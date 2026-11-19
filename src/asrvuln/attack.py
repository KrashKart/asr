"""
Gradient-based training of adversarial snippet.

General flow: Audio Tensor --> Mel Tensor --> model.forward --> Logits --> log probabilities of target (EOT)

  * ``forward``       complete suppression: maximise P(eot) at the first generated token.
  * ``forward_multi`` partial suppression: maximise P(eot) over the first ``token_limit`` tokens.
"""
import contextlib
import time
import traceback
from functools import lru_cache
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import whisper
from whisper.audio import N_SAMPLES, N_FRAMES, N_FFT, HOP_LENGTH, mel_filters
from whisper.decoding import TokenDecoder, GreedyDecoder
from whisper.tokenizer import get_tokenizer
from tqdm.auto import tqdm  # widget bars in notebooks, text bars in the terminal

from . import audio, gpu
from .config import N_MELS
from .prepare import PrepareMethod

################################
# Helpers
################################

def audio_to_mel(audio_tensor: Tensor) -> Tensor:
    """
    Audio Tensor -> Mel Tensor
    """
    return whisper.pad_or_trim(whisper.log_mel_spectrogram(audio_tensor, padding=N_SAMPLES), N_FRAMES)

def audio_to_mel_batch(audio_batch: Tensor) -> Tensor:
    """
    Batched Audio Tensor (B, N) -> Batched Mel Tensor (B, N_MELS, N_FRAMES)
    """
    if len(audio_batch.shape) == 1:
        audio_batch = audio_batch.unsqueeze(0)
    with torch.autocast(device_type=audio_batch.device.type, enabled=False):
        x = F.pad(audio_batch.float(), (0, N_SAMPLES))
        stft = torch.stft(x, N_FFT, HOP_LENGTH, window=_hann_window(x.device), return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        log_spec = torch.clamp(mel_filters(x.device, N_MELS) @ magnitudes, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.amax(dim=(-2, -1), keepdim=True) - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
    return whisper.pad_or_trim(log_spec, N_FRAMES)

@lru_cache(maxsize=None)
def _hann_window(device: torch.device) -> Tensor:
    return torch.hann_window(N_FFT, device=device)

def mel_to_logits_batch(model: whisper.model.Whisper, mel_batch: Tensor, sot_ids: Tensor) -> Tensor:
    """
    (Batched) Mel Tensor -> model.forward -> Logits Tensor
    """
    sot_ids = sot_ids.to(model.device).unsqueeze(0).expand(mel_batch.size(0), -1)
    return model.forward(mel_batch, sot_ids)

def get_loss_batch(logits: Tensor, target: int) -> Tensor:
    """
    Logits Tensor -> mean negative log probability of ``target`` over the batch
    """
    return -F.log_softmax(logits.float(), dim=-1)[:, target].mean()

def get_ids(model: whisper.model.Whisper) -> tuple:
    """
    Extract the end-of-transcript id and the start-of-sequence ids for generation and loss calculation
    """
    tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language="en", task="transcribe")
    sot_ids = torch.tensor(tokenizer.sot_sequence_including_notimestamps, requires_grad=False)
    return tokenizer.eot, sot_ids

################################
# Forward methods (losses)
################################

def forward(model: whisper.model.Whisper, snippet: Tensor, audio_batch: Tensor, prepare_method: PrepareMethod,
            sot_ids: Tensor, target_id: int) -> Tensor:
    """
    model.forward equivalent for complete suppression. Gets neg log loss of first generated token.

    sot_ids   : Start-of-sequence ids necessary to kickstart token generation.
    target_id : Target id of the token you wish to maximise probability for.

    Other arguments covered in train()
    """
    audio_batch = audio_batch.to(model.device)
    attacked_data = prepare_method(snippet, audio_batch)
    mel = audio_to_mel_batch(attacked_data)
    logits = mel_to_logits_batch(model, mel, sot_ids)[:, -1, :].squeeze(dim=1)
    return get_loss_batch(logits, target_id)

def forward_multi(model: whisper.model.Whisper, snippet: Tensor, audio_batch: Tensor,
                  decoder: TokenDecoder,
                  prepare_method: PrepareMethod,
                  sot_ids: Tensor, target_id: int,
                  token_limit: int = 5) -> Tensor:
    """
    model.forward equivalent for partial suppression. Gets mean neg log loss of first {token_limit} generated tokens.

    decoder     : TokenDecoder instance from the Whisper library. You can customise search options for the decoder as well.
    sot_ids     : Start-of-sequence ids necessary to kickstart token generation.
    target_id   : Target id of the token you wish to maximise probability for.
    token_limit : Number of tokens to allow model to generate before forcing end-of-sequence token.

    Other arguments covered in train()
    """
    audio_batch = audio_batch.to(model.device)
    attacked_data = prepare_method(snippet, audio_batch)
    mel = audio_to_mel_batch(attacked_data)
    batch_size = mel.size(0)

    # cache cross KV to avoid recomputing for every token generated.
    audio_features = model.embed_audio(mel)
    cross_kv = _cross_attention_kv(model, audio_features)

    tokens = sot_ids.to(model.device).unsqueeze(0).expand(batch_size, -1)
    sum_logprobs = torch.zeros(batch_size, device=model.device)
    final_loss = torch.zeros(batch_size, device=model.device)

    for _ in range(token_limit):
        alive = tokens[:, -1] != target_id
        logits = _next_token_logits(model, tokens, audio_features, cross_kv)
        final_loss = final_loss + F.log_softmax(logits.float(), dim=-1)[:, target_id] * alive
        tokens, completed = decoder.update(tokens, logits, sum_logprobs)
        if completed:
            break

    return (-final_loss / token_limit).mean()

def _cross_attention_kv(model: whisper.model.Whisper, audio_features: Tensor) -> dict:
    """
    Cross-attention KV for every decoder block. Computed once per batch.
    """
    kv = {}
    for block in model.decoder.blocks:
        attn = block.cross_attn
        kv[attn.key] = attn.key(audio_features)
        kv[attn.value] = attn.value(audio_features)
    return kv

def _next_token_logits(model: whisper.model.Whisper, tokens: Tensor, audio_features: Tensor, cross_kv: dict) -> Tensor:
    """
    Get next token logits.
    """
    dec = model.decoder
    x = dec.token_embedding(tokens) + dec.positional_embedding[: tokens.shape[-1]]
    x = x.to(audio_features.dtype)
    for block in dec.blocks:
        x = block(x, audio_features, mask=dec.mask, kv_cache=cross_kv)
    x = dec.ln(x[:, -1, :])
    return (x @ dec.token_embedding.weight.to(x.dtype).T).float()

def _to_cpu(t: Tensor) -> Tensor:
    """Detached CPU copy (always a copy, even when ``t`` is already on the CPU)."""
    return t.detach().to("cpu", copy=True)

def _needs_decoder(forward_method: Callable) -> bool:
    return forward_method is forward_multi or forward_method.__name__ in ("forward_multi", "forward_auto")

def _transcript_length(model: whisper.model.Whisper, attacked: Tensor) -> int:
    return len(model.transcribe(attacked.squeeze(), language="en", condition_on_previous_text=False, fp16=True)["text"])

################################
# Training loop
################################

def train(model: whisper.model.Whisper, forwardMethod: Callable,
          train_data: DataLoader, valid_data: DataLoader,
          prepare_method: PrepareMethod,
          writer: Optional[SummaryWriter] = None,
          train_success: Optional[dict] = None, valid_success: Optional[dict] = None,
          lr: float = 1e-3,
          iter_limit: Optional[int] = None, mins_limit: Optional[int] = None, patience: Optional[int] = None,
          clamp_epsilon: Optional[float] = None, gap: float = 0.0, amp: bool = False) -> tuple:
    """
    Train an adversarial snippet.

    Keyboard interrupts stop training cleanly and returns best snippet so far.

    Arguments:
        model          : Whisper Model to train on.
        forwardMethod  : Either forward (complete suppression) or forward_multi (partial suppression).
        train_data     : Train data mounted to PyTorch DataLoader, yielding (audio, idx) (see data.collate_idx).
        valid_data     : Validation data mounted to PyTorch DataLoader, yielding (audio, idx).
        prepare_method : method of applying snippet to audio. Must be a PrepareMethod (prepare.py / defences.py).
        writer         : Tensorboard writer. If None, doesn't write to Tensorboard.
        train_success  : Dict to track transcript length per training example per iteration. If None, does not track.
        valid_success  : Dict to track transcript length per validation example per iteration. If None, does not track.
        lr             : Learning rate for optimizer. Recommended 1e-3.
        iter_limit     : Iteration limit to halt training. If None, does not halt training on iters.
        mins_limit     : Time limit in minutes to halt training. If None, does not halt training on time.
        patience       : Patience for early stopping. If Validation Loss increases or decreases by gap (see gap parameter) for {patience} iterations in a row, then stops training.
        clamp_epsilon  : Clamp Limits for the snippet. If None, does not clamp snippet at all.
        gap            : The minimum validation loss difference between iterations to be classified as "convergence", i.e. if loss_i and loss_(i-1) differ by {gap}, then decrease patience/terminate training.
        amp            : Run the Whisper forward/backward under bf16 autocast (highly recommended, minimal loss of precision).
    
    Returns:
        best_snippet   : Best snippet judged by validation loss (CPU)
        snippets       : list of snippets across training iterations (CPU), starting with the initial snippet
        train_success  : Dictionary of success across iterations for each training example. If None, returns None.
        valid_success  : Dictionary of success across iterations for each validation example. If None, returns None.
    """
    torch.autograd.set_detect_anomaly(False)
    num_training_batches = len(train_data)
    num_valid_batches = len(valid_data)

    time_limit = mins_limit * 60 if mins_limit else None

    # prepare snippet
    with torch.no_grad():
        snippet = torch.rand(prepare_method.snippet_size, device=model.device)
        if clamp_epsilon:
            snippet *= clamp_epsilon
    snippet.requires_grad_(True)

    # snippets for tracking progress
    snippets = [_to_cpu(snippet)]
    best_snippet = snippets[0].clone()

    # optimizer and scheduler, loss, patience
    optim = torch.optim.AdamW([snippet], lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)
    lowest_valid_loss = float("inf")
    curr_patience = patience

    # set up decoder if needed, and get the target and sot ids for loss calculation
    target_id, sot_ids = get_ids(model)
    sot_ids = sot_ids.to(model.device)
    decoder = None
    if _needs_decoder(forwardMethod):
        decoder = GreedyDecoder(0.0, eot=target_id)

    # save the original requires_grad of the model parameters so can restore later
    param_grad_state = [p.requires_grad for p in model.parameters()]

    # CUDA settings
    use_amp = amp and model.device.type == "cuda"
    non_blocking = model.device.type == "cuda"

    # helper functions
    def run_forward(batch):
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
        with ctx:
            if decoder:
                return forwardMethod(model, snippet, batch, decoder, prepare_method, sot_ids, target_id)
            return forwardMethod(model, snippet, batch, prepare_method, sot_ids, target_id)

    def track_success(success: dict, batch: Tensor, idx):
        for row, i in zip(batch, idx if isinstance(idx, list) else [idx]):
            seq_length = _transcript_length(model, prepare_method(snippet, row.unsqueeze(0)))
            success[i] = success.get(i, []) + [seq_length]

    # display attack method and snippet for sanity check
    print(f"Prepare method: {prepare_method.name}")
    print(f"Snippet initialised to [{torch.min(snippet)}, {torch.max(snippet)}] of size {prepare_method.snippet_size}")
    print(f"Clamp: {clamp_epsilon}\nTime Limit (Mins): {mins_limit}\nEpochs Limit: {iter_limit}")
    print(f"Tracking training success: {train_success is not None}\nTracking valid success: {valid_success is not None}")

    if writer:
        writer.add_image("Attack Snippet", audio.mel_image(snippet.detach().to("cpu")), 0, dataformats="HWC")
        writer.flush()

    bar_kw = dict(leave=False, unit="batch", dynamic_ncols=True)
    loss_refresh = 10  # update the running-loss readout every N batches (each readout is one GPU sync)
    itera = 0
    stop_reason = None
    pbar = None
    loss = batch = v = avg_training_loss = avg_valid_loss = None
    base_cuda_usage = gpu.get_cuda_usage()

    try:
        model.requires_grad_(False)

        # 2 loading bars:
        # outer bar: postfix = latest epoch summary.
        # inner bars: batches within the current epoch's train / validation pass, with rate + ETA.
        pbar = tqdm(total=iter_limit, desc="Epochs", unit="epoch", dynamic_ncols=True)
        start_time = time.monotonic()

        while True:
            if iter_limit and itera >= iter_limit:
                stop_reason = f"Epoch limit ({iter_limit}) reached"
                break
            if time_limit and time.monotonic() - start_time >= time_limit:
                stop_reason = f"Time limit ({mins_limit} min) reached"
                break
            if patience and curr_patience <= 0:
                stop_reason = f"Patience ({patience}) expired"
                break
            if clamp_epsilon and audio.violates_clamp(snippet, clamp_epsilon):
                raise ValueError("Snippet values violate clamp constraint!!")
            itera += 1

            # accumulated on the GPU so there's no host sync per batch
            avg_training_loss = torch.zeros((), device=model.device)
            total_cuda_usage_iter = 0

            with tqdm(train_data, total=num_training_batches, desc=f"Epoch {itera} train", **bar_kw) as bar:
                for batch_no, (batch, idx) in enumerate(bar):
                    batch = batch.to(model.device, non_blocking=non_blocking)
                    loss = run_forward(batch)

                    total_cuda_usage_iter += gpu.get_cuda_usage()
                    avg_training_loss += loss.detach()

                    loss.backward()
                    optim.step()
                    optim.zero_grad(set_to_none=True)

                    if clamp_epsilon:
                        with torch.no_grad():
                            snippet.clamp_(min=-clamp_epsilon, max=clamp_epsilon)

                    if train_success is not None:
                        with torch.no_grad():
                            track_success(train_success, batch, idx)

                    if (batch_no + 1) % loss_refresh == 0:
                        bar.set_postfix(loss=f"{avg_training_loss.item() / (batch_no + 1):.4f}", refresh=False)

            snippets.append(_to_cpu(snippet))
            avg_training_loss = (avg_training_loss / num_training_batches).item()

            # validation
            avg_valid_loss = torch.zeros((), device=model.device)
            num_valid_examples = 0
            with torch.no_grad(), tqdm(valid_data, total=num_valid_batches, desc=f"Epoch {itera} valid", **bar_kw) as bar:
                for v, idx in bar:
                    v = v.to(model.device, non_blocking=non_blocking)
                    # weight by batch size so a smaller final batch doesn't skew the per-example mean
                    avg_valid_loss += run_forward(v) * v.size(0)
                    num_valid_examples += v.size(0)

                    if valid_success is not None:
                        track_success(valid_success, v, idx)
            avg_valid_loss = (avg_valid_loss / num_valid_examples).item()

            # track lowest valid loss and save the snippet with lowest valid loss
            improved = not (-gap <= avg_valid_loss - lowest_valid_loss <= gap or avg_valid_loss > lowest_valid_loss)
            if improved:
                curr_patience = patience
                lowest_valid_loss = avg_valid_loss
                best_snippet = snippets[-1].clone()
            elif patience:
                curr_patience -= 1

            lr_now = scheduler.get_last_lr()[0]
            pbar.write(f"Epoch {itera:>3} | train {avg_training_loss:.4f} | valid {avg_valid_loss:.4f}"
                       f"{' *best*' if improved else ''} | lr {lr_now:.1e}"
                       + (f" | patience {curr_patience}/{patience}" if patience else ""))
            pbar.set_postfix(train=f"{avg_training_loss:.4f}", valid=f"{avg_valid_loss:.4f}",
                             best=f"{lowest_valid_loss:.4f}", refresh=False)
            pbar.update(1)

            if writer:
                writer.add_scalar("Training average loss", avg_training_loss, itera)
                writer.add_scalar("Validation average loss", avg_valid_loss, itera)
                if torch.cuda.is_available():
                    writer.add_scalar("GPU RAM Usage", total_cuda_usage_iter / num_training_batches - base_cuda_usage, itera)
                writer.add_image("Attack Snippet", audio.mel_image(snippet.detach().to("cpu")), itera, dataformats="HWC")
                writer.flush()

            scheduler.step()

    except KeyboardInterrupt:
        # just in case lol
        stop_reason = "Interrupted by user"

    except Exception as e:
        # keep the best snippet found so far
        stop_reason = f"Failed with {type(e).__name__}"
        if pbar is not None:
            pbar.close()
        traceback.print_exc()
        if isinstance(e, torch.cuda.OutOfMemoryError):
            tqdm.write("CUDA out of memory: try a smaller --batch-size / --valid-batch-size, or --amp.")

    finally:
        if pbar is not None:
            pbar.close()

        for p, requires_grad in zip(model.parameters(), param_grad_state):
            p.requires_grad_(requires_grad)

        if writer:
            writer.flush()

        # release every GPU reference then hand the memory back to the driver.
        loss = batch = v = avg_training_loss = avg_valid_loss = None
        optim.state.clear()
        snippet.grad = None
        del optim, scheduler, snippet, run_forward, track_success
        gpu.cleanup()

    epochs_done = pbar.n if pbar is not None else 0
    tqdm.write(f"{stop_reason or 'Stopped'} after {epochs_done} completed epoch(s). Best valid loss: {lowest_valid_loss:.4f}")
    return best_snippet, snippets, train_success, valid_success

def success_to_tensor(success: dict) -> tuple:
    """
    ``train``'s success dict ``{example idx: [transcript length per iteration]}`` ->
    ``(ids, lengths)``: ids (n_examples,) sorted, and lengths (n_examples, n_iterations), as float.

    This is the on-disk format of ``outputs/tensors/<mode>_<split>_success.pt`` / ``<mode>_<split>_ids.pt`` (see
    ``viz.plot_success_heatmap``). If training stopped partway through an epoch, some examples
    have one more entry than others; the gaps are NaN, which the heatmap leaves blank.
    """
    ids = sorted(success)
    width = max((len(v) for v in success.values()), default=0)
    lengths = torch.full((len(ids), width), float("nan"))
    for row, i in enumerate(ids):
        lengths[row, :len(success[i])] = torch.tensor(success[i], dtype=torch.float)
    return torch.tensor(ids), lengths

def stack_history(best_snippet: Tensor, snippets: list) -> Tensor:
    """
    Stack the per-iteration snippets with the best snippet *last*: shape (iters + 1, N).
    This is the on-disk format used for ``outputs/snippets/*.pt`` (``saved[-1]`` is the best).
    """
    return torch.stack([s.detach().cpu() for s in snippets] + [best_snippet.detach().cpu()]).squeeze(1)
