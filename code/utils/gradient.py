import gc, traceback, datetime
import whisper
from whisper.audio import N_SAMPLES, N_FRAMES
from whisper.decoding import TokenDecoder, GreedyDecoder
from whisper.tokenizer import get_tokenizer

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import bleu_score

from . import audio, gpu
from .attacks import PrepareMethod

from typing import Optional
from tqdm import tqdm

################################
# Helpers
# General flow is Audio Tensor --> Mel Tensor --> model.forward --> Logits --> Get log probabilities
################################

def audio_to_mel(audio: Tensor) -> Tensor:
    return whisper.pad_or_trim(whisper.log_mel_spectrogram(audio, padding=N_SAMPLES),
                              N_FRAMES)

def audio_to_mel_batch(audio_batch: Tensor) -> Tensor:
    if len(audio_batch.shape) == 1:
        audio_batch = audio_batch.unsqueeze(0)
    return torch.stack([audio_to_mel(audio) for audio in audio_batch])

def mel_to_logits_batch(model: whisper.model.Whisper, mel_batch: Tensor, sot_ids: Tensor) -> Tensor:
    sot_ids = sot_ids.unsqueeze(0).expand(mel_batch.size(0), -1).to(model.device)
    return model.forward(mel_batch, sot_ids)

def get_loss_batch(logits: Tensor, target: Tensor) -> Tensor:
    sf = torch.nn.Softmax(dim=1)
    log_probs = torch.log(sf(logits))
    tgt_probs = log_probs[:,target].squeeze()
    return -1 * torch.mean(tgt_probs)

def _get_ids(model: whisper.model.Whisper) -> tuple:
    tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language="en", task="transcribe")
    sot_ids = torch.tensor(tokenizer.sot_sequence_including_notimestamps, requires_grad=False)
    return tokenizer.eot, sot_ids

################################
# Train, validation, test
################################

def train(model: whisper.model.Whisper, forwardMethod,
            train_data: DataLoader, valid_data: DataLoader,
            prepare_method: PrepareMethod,
            writer: Optional[SummaryWriter] = None,
            train_success: Optional[dict] = None, valid_success: Optional[dict] = None,
            lr: float = 1e-3,
            iter_limit: Optional[int] = None, mins_limit: Optional[int] = None, patience: Optional[int] = None, clamp_epsilon: Optional[float] = None) -> torch.Tensor:
    
    torch.autograd.set_detect_anomaly(False)
    loss = torch.tensor(float("inf"), requires_grad=True)
    num_training_batches = len(train_data)
    num_valid_batches = len(valid_data)

    time_limit = mins_limit * 60 if mins_limit else None

    snippet = torch.rand(prepare_method.snippet_size, requires_grad=True, device=model.device) # adversarial snippet
    
    snippets = [snippet]
    buffer = None

    if clamp_epsilon:
        with torch.no_grad():
            snippet = snippet * clamp_epsilon
    snippet.requires_grad = True

    optim = torch.optim.AdamW([snippet], lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)

    attack_stack, attacked_data, mel, logits, pbar, avg_valid_loss, decoder = None, None, None, None, None, None, None
    lowest_valid_loss = torch.tensor(float("inf"))
    curr_patience = patience
    best_snippet = snippet.detach().clone()
    
    target_id, sot_ids = _get_ids(model)
    
    if forwardMethod.__name__ == "forward_auto":
        decoder = GreedyDecoder(0.0, eot=target_id)

    # display attack method and snippet for sanity check
    print(f"Prepare method: {prepare_method.name}")
    print(f"Snippet initialised to [{torch.min(snippet)}, {torch.max(snippet)}] of size {prepare_method.snippet_size}")
    print(f"Clamp: {clamp_epsilon}\nTime Limit (Mins): {mins_limit}\nEpochs Limit: {iter_limit}")
    print(f"Tracking training success: {train_success is not None}\nTracking valid success: {valid_success is not None}")

    # log attack snippet
    if writer:
        writer.add_image("Attack Snippet", audio.mel_image(snippet.detach().to("cpu")), 0, dataformats="HWC")
        writer.flush()

    # progress bar
    pbar = tqdm(range(1), desc="Training", ncols=0)
    itera = 0

    # track gpu usage
    base_cuda_usage = gpu.get_cuda_usage()

    try:
        while True:
            buffer = torch.zeros((len(train_data), 1))
            itera += 1
            if iter_limit:
                iter_limit -= 1
                if iter_limit <= 0:
                    pbar.set_postfix_str("Epoch limit reached! Terminating...")
                    break
            if time_limit and pbar.format_dict["elapsed"] >= time_limit:
                pbar.set_postfix_str("Time limit reached! Terminating...")
                break
            if patience and avg_valid_loss:
                if curr_patience <= 0:
                    pbar.set_postfix_str("Patience expired! Terminating...")
                    break
            if clamp_epsilon and audio.violates_clamp(snippet, clamp_epsilon):
                raise ValueError("Snippet values violate clamp constraint!!")

            avg_training_loss = 0
            total_cuda_usage_iter = 0
            
            delta = snippet.clone().detach()
            
            for batch_no, (batch, idx) in enumerate(train_data):
                pbar.set_postfix_str(f"Iter {itera}, Training Batch {batch_no + 1}/{num_training_batches}")

                batch = batch.to(model.device)
                if decoder:
                    loss = forwardMethod(model, snippet, batch, decoder, prepare_method, sot_ids, target_id)
                else:
                    loss = forwardMethod(model, snippet, batch, prepare_method, sot_ids, target_id)
                
                # get training metrics
                total_cuda_usage_iter += gpu.get_cuda_usage()
                avg_training_loss += loss.detach().item()
                
                # backprop
                loss.backward()
                
                optim.step()
                optim.zero_grad()
                
                # clamp snippet
                if clamp_epsilon:
                    with torch.no_grad():
                        snippet.clamp_(min=-clamp_epsilon, max=clamp_epsilon)
                        
                snippet.requires_grad = True
                batch.to("cpu")
                
                if train_success:
                    with torch.no_grad():
                        seq_length = len(model.transcribe(attacked_data.squeeze(), language="en", condition_on_previous_text=False, fp16=True)["text"])
                        train_success[idx] = train_success.get(idx, []) + [seq_length]
            
            snippets.append(snippet.detach().clone())
            avg_training_loss /= len(train_data)

            # validation
            avg_valid_loss = 0
            with torch.no_grad():
                for batch_no, (v, idx) in enumerate(valid_data):
                    pbar.set_postfix_str(f"Iter {itera}, Validation Batch {batch_no + 1}/{num_valid_batches}")
                    if decoder:
                        avg_valid_loss += forwardMethod(model, snippet, v, decoder, prepare_method, sot_ids, target_id)
                    else:
                        avg_valid_loss += forwardMethod(model, snippet, v, prepare_method, sot_ids, target_id)
                    
                    if valid_success:
                        seq_length = len(model.transcribe(attacked_data_valid.squeeze(), language="en", condition_on_previous_text=False, fp16=True)["text"])
                        valid_success[idx] = valid_success.get(idx, []) + [seq_length] 
            avg_valid_loss /= num_valid_batches
            avg_valid_loss = avg_valid_loss.item()
            
            # track lowest valid loss and save the snippet with lowest valid loss
            if avg_valid_loss >= lowest_valid_loss:
                curr_patience -= 1
            else:
                curr_patience = patience
                lowest_valid_loss = avg_valid_loss
                best_snippet = snippet.detach().clone()

            pbar.write(f"Trng Avg Loss: {avg_training_loss} | Valid Avg Loss: {avg_valid_loss} | Patience: {curr_patience} | LR: {scheduler.get_last_lr()} | Epoch Limit: {iter_limit}")

            if writer:
              # log training and validation losses
                writer.add_scalar("Training average loss", avg_training_loss, itera)
                writer.add_scalar("Validation average loss", avg_valid_loss, itera)

                # log GPU RAM usage
                if torch.cuda.is_available():
                    writer.add_scalar("GPU RAM Usage", total_cuda_usage_iter / num_training_batches - base_cuda_usage, itera)

              # log attack snippet and flush
                writer.add_image("Attack Snippet", audio.mel_image(snippet.detach().to("cpu")), itera, dataformats="HWC")
                writer.flush()
            
            # LR decay
            scheduler.step()
            
            # refresh pbar to (hopefully) force update of progress bar
            pbar.refresh()

    except Exception as e:
        # need to explicitly close pbar here so traceback can print error
        if pbar is not None:
            pbar.clear()
            pbar.close()
            traceback.print_exc()

    finally:
        # close pbar to free stdout/stdsys (cant rmb which one)
        if pbar is not None:
            pbar.clear()
            pbar.close()

        # clear tensors from GPU memory to
        # prevent memory leak
        if attacked_data is not None:
            attacked_data.to("cpu")
            del attacked_data
            print("Cleared attacked data")

        if attack_stack is not None:
            attack_stack.to("cpu")
            del attack_stack
            print("Cleared attack stack")

        if mel is not None:
            mel.to("cpu")
            del mel
            print("Cleared mel")

        if logits is not None:
            logits.to("cpu")
            del logits
            print("Cleared logits")
        
        if buffer is not None:
            buffer.to("cpu")
            del buffer
            print("Cleared buffer")

        loss.to("cpu")
        del loss
        print("Cleared loss")

        # empty GPU cache and garbage collect
        gpu.cleanup()
        
        return best_snippet.detach().to("cpu"), snippets, train_success, valid_success

def forward(model: whisper.model.Whisper, snippet: Tensor, audio: Tensor, prepare_method: PrepareMethod, sot_ids: Tensor, target_id: int) -> Tensor:
    audio = audio.to(model.device)
    attacked_data = prepare_method(snippet, audio)
    mel = audio_to_mel_batch(attacked_data)
    logits = mel_to_logits_batch(model, mel, sot_ids)[:,-1,:].squeeze(dim=1)
    loss = get_loss_batch(logits, target_id)
    return loss

def forward_auto(model: whisper.model.Whisper, snippet: Tensor, audio: Tensor, 
                 decoder: TokenDecoder, 
                 prepare_method: PrepareMethod, 
                 sot_ids: Tensor, target_id: int,
                 token_limit: int = 10) -> Tensor:
    
    sum_logprobs = torch.tensor([0.0], device=model.device)
    sf = torch.nn.Softmax(dim=1)
    tokens = sot_ids.unsqueeze(0).to(model.device)
    final_loss = 0.0
    completed = False
    running_sum = sum([t for t in range(1, token_limit)])
    
    audio = audio.to(model.device)
    attacked_data = prepare_method(snippet, audio)
    mel = audio_to_mel_batch(attacked_data)
    
    assert mel.device == tokens.device, f"Mel device: {mel.device}, Tokens device: {tokens.device}"
    
    counter = 0
    while not completed and counter < token_limit:
        logits = model.forward(mel, tokens)[:, -1, :]
        final_loss += torch.log(sf(logits)[:, target_id]) * ((token_limit - counter) / running_sum)
        tokens, completed = decoder.update(tokens, logits, sum_logprobs)
        counter += 1

    return -final_loss

def evaluate(model: whisper.model.Whisper, snippet: Tensor, prepare_method: PrepareMethod, test_dataset: DataLoader, clamp_ep: float, position: tuple):
    print(f"Clamp: {clamp_ep}\nPrepare Method: {prepare_method.name}\nSnippet Size: {prepare_method.snippet_size}\nPosition: {position}")
    empty_counter = 0
    char_counter = 0
    total_examples = 0
    original_chars = 0
    non_empty = 0
    avg_bleu_score = 0

    snippet = snippet.to(model.device)
    pbar = tqdm(range(len(test_dataset)), desc="Inference")
    test_dataset_iter = iter(test_dataset)
    model.eval()

    for i in pbar:
        # evaluate if there are any words at all
        example, answer = next(test_dataset_iter).values()
        if isinstance(answer, tuple) or isinstance(answer, list):
            answer = answer[0]
        if answer != "ignore_time_segment_in_scoring":
            attacked_example = prepare_method(snippet, example.to(model.device))
            transcription = model.transcribe(attacked_example.squeeze(), language="en", condition_on_previous_text=False, fp16=True)["text"].strip()

            if not transcription:
                empty_counter += 1
            else:
                non_empty += 1
                avg_bleu_score += bleu_score(transcription, [answer], n_gram=1)
            char_counter += len(transcription.strip())
            original_chars += len(answer)
            total_examples += 1
            pbar.set_postfix_str(f"Valid Examples: {total_examples} | Empty Sequences: {empty_counter} | Total SL: {char_counter} | Non-empty ASL: {'Undefined' if not non_empty else char_counter / non_empty} | Total Bleu Score: {avg_bleu_score}")

        example.to("cpu")

    pbar.close()
    print("\n")
    print(f"Total valid examples: {total_examples}")
    print(f"Success rate (Empty): {empty_counter/total_examples}")
    print(f"Success rate (ASL): {char_counter/total_examples} (attacked) out of {original_chars/total_examples} (original)")
    print(f"Average Bleu Score: {'Undefined' if not non_empty else avg_bleu_score / non_empty}")