"""
Attention extraction for HuggingFace Whisper.

Shapes produced by the stacking helpers (tiny.en: 4 blocks x 6 heads):
    stack_encoder_attns -> (blocks, heads, enc_tokens, enc_tokens)
    stack_cross_attns   -> (blocks, heads, dec_tokens, enc_tokens)
    stack_decoder_attns -> (blocks, heads, dec_tokens, dec_tokens)
"""
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import ENCODER_FPS, SAMPLE_RATE


def inference(audio_tensor: Tensor, model, processor, skip_special_tokens: bool = False) -> tuple:
    """
    Pass audio to model for inference. Audio tensor must be non-batched (single example) and 1-Dimensional.

    Returns (decoded, res, timestamps, encoder_attentions, decoder_attentions, cross_attentions)
    """
    assert audio_tensor.dim() == 1, f"Audio tensor must be 1-Dimensional! Got {audio_tensor.dim()} dims"

    inputs = processor(audio_tensor, return_tensors="pt", sampling_rate=SAMPLE_RATE)
    input_features = inputs.input_features.to(model.device)

    res = model.generate(input_features, return_token_timestamps=True, output_attentions=True,
                         output_hidden_states=True, return_dict_in_generate=True)
    
    sequences = res["sequences"].squeeze()
    decoded = processor.decode(sequences, skip_special_tokens=skip_special_tokens)

    list_of_tokens = [processor.decode([int(r)], skip_special_tokens=skip_special_tokens) for r in sequences]
    timestamps = list(zip(list_of_tokens, res["token_timestamps"].squeeze().tolist()))

    return (decoded, res, timestamps, res["encoder_attentions"], res["decoder_attentions"], res["cross_attentions"])


def encoder_limit(n_samples: int) -> int:
    """Number of encoder tokens that cover ``n_samples`` of real audio (the rest is 30 s padding)."""
    return int(n_samples / SAMPLE_RATE * ENCODER_FPS) + 1


def stack_encoder_attns(e_attns: Sequence[Tensor], limit: Optional[int] = None) -> Tensor:
    return torch.stack(list(e_attns)).squeeze()[:, :, :limit, :limit]


def stack_cross_attns(c_attns: Sequence[Sequence[Tensor]], limit: Optional[int] = None) -> Tensor:
    """Stack generate()'s per-step cross attentions (skipping the prompt step) and crop the encoder axis."""
    return torch.stack([torch.stack(c).squeeze() for c in c_attns[1:]]).permute(1, 2, 0, 3)[:, :, :, :limit]


def stack_decoder_attns(d_attns: Sequence[Sequence[Tensor]]) -> Tensor:
    """Stack generate()'s per-step decoder self-attentions, right-padding each step to the final length."""
    final_len = d_attns[-1][-1].size(3)
    steps = [F.pad(torch.stack(d).squeeze(), (0, final_len - d[-1].size(3), 0, 0), "constant", 0.0) for d in d_attns[1:]]
    return torch.stack(steps).permute(1, 2, 0, 3)


def attention_over_snippets(model, processor, snippets: Tensor, example: Tensor, kind: str = "cross",
                            block: int = 0, head: int = 2, pad_value: float = -0.05) -> Tensor:
    """
    Attention map of one (block, head) for ``example`` prefixed by each snippet in ``snippets``
    (e.g. the snippet at every training iteration). Returns (n_snippets, rows, cols); cross
    attention maps are bottom-padded with ``pad_value`` to the longest decoded length.
    """
    assert kind in ("cross", "encoder"), f"kind must be 'cross' or 'encoder', got {kind!r}"
    limit = encoder_limit(example.size(0) + snippets.size(1))
    maps = []
    for s in snippets.to(example.device):
        _, _, _, e_attns, _, c_attns = inference(torch.cat([s, example]), model, processor, skip_special_tokens=True)
        stacked = stack_cross_attns(c_attns, limit) if kind == "cross" else stack_encoder_attns(e_attns, limit)
        maps.append(stacked[block, head])
    max_len = max(m.size(0) for m in maps)
    maps = [F.pad(m, (0, 0, 0, max_len - m.size(0)), value=pad_value) for m in maps]
    return torch.stack(maps)


def get_spikes(attn: Tensor, lim: float) -> tuple:
    """
    Encoder token indices whose (min-max normalised) attention is >= lim, and the
    corresponding audio *sample* positions (each encoder token is 0.02 s).
    """
    maxi, mini = attn.max(), attn.min()
    normalised_attn = (attn - mini) / (maxi - mini)
    masked = torch.where(normalised_attn >= lim, 1.0, 0.0)
    indices = masked.nonzero()[:, 1].unique().cpu()
    samples = indices * (SAMPLE_RATE // ENCODER_FPS)
    return indices, samples


def smoothen(frames: Tensor, factor: int = 5) -> Tensor:
    """
    Linearly interpolate (factor - 1) frames between consecutive frames, for smooth animations
    """
    result = []
    for i in range(frames.size(0)):
        if i != 0:
            step_diff = (frames[i, :, :] - frames[i - 1, :, :]) / (factor - 1)
            for j in range(factor - 1):
                result.append(frames[i - 1, :, :] + j * step_diff)
        result.append(frames[i, :, :])
    assert len(result) == (frames.size(0) - 1) * factor + 1
    return torch.stack(result)
