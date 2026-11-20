"""
Causal Mediation Analysis, inspired by ROME (arXiv:2202.05262).

1. Clean run: record every decoder block's output at every generation step.
2. Corrupted run: add Gaussian noise to the decoder token embeddings.
3. Corrupted run with correction: restore one block's clean output at one step.

Effects on P(EOT):  TE = clean - corrupt,  DE = clean - corrected,  IE = corrected - corrupt
"""
import traceback
from typing import Any, Callable, Dict, List, Union

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from transformers import WhisperForConditionalGeneration
from tqdm import tqdm

from ..config import SAMPLE_RATE

IntTensorDict = Dict[int, List[Tensor]]
Number = Union[int, float]
Whisper = WhisperForConditionalGeneration
Hook = Callable[..., Any]

#####################################
# Bookkeeping
#####################################

def reset(*args: Union[List[RemovableHandle], IntTensorDict]) -> None:
    """Remove all hooks in the given lists and clear the given lists / dicts."""
    for container in args:
        if isinstance(container, list):
            for handle in container:
                handle.remove()
        container.clear()
        assert not container, f"{container} cannot be cleared!"

def append_dict(d: IntTensorDict, key: int, value: Tensor) -> None:
    d.setdefault(key, []).append(value)

def remove_hook_list(hook_list: List[RemovableHandle]) -> None:
    for h in hook_list:
        h.remove()
    hook_list.clear()

#####################################
# Hooks
#####################################

def clean_hook(idx: int, activations: IntTensorDict):
    def hook(module, args, output):
        append_dict(activations, idx, output)
        return output
    return hook

def corrupt_embed_hook(alpha: Number = 1):
    def hook(module, args, output):
        epsilons = torch.normal(0, alpha * torch.std(output.squeeze()).item(), size=tuple(output.shape)).to(output.device)
        return output + epsilons
    return hook

def correction_hook(block: int, token: int, activations: IntTensorDict):
    activation_count = 0
    def hook(module, args, out):
        nonlocal activation_count
        if activation_count == token:
            out = activations[block][token]
        activation_count += 1
        return out
    return hook

#####################################
# Registration
#####################################

def register_all_decoder_blocks(model: Whisper, hook_creator: Hook, activations: IntTensorDict, hl: List[RemovableHandle]) -> None:
    for i, block in enumerate(model.model.decoder.layers):
        hl.append(block.register_forward_hook(hook_creator(i, activations)))

def register_at_decoder_block(model: Whisper, hook_creator: Hook, block: int, hl: List[RemovableHandle], *args: Any) -> RemovableHandle:
    h = model.model.decoder.layers[block].register_forward_hook(hook_creator(*args))
    hl.append(h)
    return h

def register_correction_hook(model: Whisper, block: int, token: int, activations: IntTensorDict, hl: List[RemovableHandle]) -> RemovableHandle:
    return register_at_decoder_block(model, correction_hook, block, hl, block, token, activations)

def register_encoder_embedding_hook(model: Whisper, hook_creator: Hook, hl: List[RemovableHandle], alpha: Number = 1) -> RemovableHandle:
    h = model.model.encoder.embed_positions.register_forward_hook(hook_creator(alpha))
    hl.append(h)
    return h

def register_decoder_embedding_hook(model: Whisper, hook_creator: Hook, hl: List[RemovableHandle], alpha: Number = 1) -> RemovableHandle:
    h = model.model.decoder.embed_tokens.register_forward_hook(hook_creator(alpha))
    hl.append(h)
    return h

#####################################
# Inference
#####################################

def infer_with_probs(audio: Tensor, model: Whisper, processor) -> tuple:
    """Generate a transcription and return (decoded, res, P(EOT) at every generation step)."""
    eos_id = processor.tokenizer.eos_token_id
    inputs = processor(audio, return_tensors="pt", sampling_rate=SAMPLE_RATE)
    input_features = inputs.input_features.to(model.device)

    res = model.generate(input_features, return_dict_in_generate=True, output_scores=True)
    eos_probs = torch.tensor([torch.nn.Softmax(dim=0)(s.squeeze())[eos_id] for s in res["scores"]])

    decoded = processor.decode(res["sequences"].squeeze(), skip_special_tokens=True)
    return decoded, res, eos_probs

def run_causal_trace(clean_model: Whisper, corrupted_model: Whisper, processor, wave: Tensor,
                     iters: int = 10, alpha: Number = 3) -> Dict[str, Tensor]:
    """
    Run the corrupt / restore experiment ``iters`` times (fresh noise each time) on one example and
    average the effects. ``clean_model`` and ``corrupted_model`` must be two separate model instances.

    Returns a dict of (tokens, blocks) tensors: "total", "direct", "indirect", plus the
    last iteration's "clean" and "corrupt" P(EOT) for reference.
    """
    n_blocks = len(clean_model.model.decoder.layers)
    activations: IntTensorDict = {}
    debug_activations: IntTensorDict = {}
    hooks: List[RemovableHandle] = []
    temp_hooks: List[RemovableHandle] = []
    results: List[Tensor] = []
    total_effect = direct_effect = indirect_effect = 0
    clean_probs = corrupt_probs = None
    correction = None

    pbar = tqdm(range(iters), leave=True, ncols=0, desc="Aggregate")
    try:
        for _ in pbar:
            results.clear()
            reset(activations, debug_activations, hooks, temp_hooks)
            register_all_decoder_blocks(clean_model, clean_hook, activations, hooks)

            # collect activations and find clean probs
            pbar.set_postfix_str("Clean run")
            _, _, clean_probs = infer_with_probs(wave, clean_model, processor)
            clean_probs = clean_probs.unsqueeze(1).repeat(1, n_blocks)
            assert activations, "Activations is empty! Reverify"

            # corrupt model and find corrupt probs
            pbar.set_postfix_str("Corrupted run")
            register_decoder_embedding_hook(corrupted_model, corrupt_embed_hook, hooks, alpha=alpha)
            _, _, corrupt_probs = infer_with_probs(wave, corrupted_model, processor)
            corrupt_probs = corrupt_probs.unsqueeze(1).repeat(1, n_blocks)
            corrupt_probs = torch.nn.functional.pad(corrupt_probs, (0, 0, 0, clean_probs.size(0) - corrupt_probs.size(0)), value=1.0)

            # sequentially restore each block at each token
            n_tokens = len(activations[0])
            for token in range(n_tokens):
                pbar.set_postfix_str(f"Corrupted run with correction: {token + 1}/{n_tokens}")
                blocks = []
                for block in range(n_blocks):
                    correction = register_correction_hook(corrupted_model, block, token, activations, temp_hooks)
                    _, _, correction_probs = infer_with_probs(wave, corrupted_model, processor)
                    correction.remove()
                    blocks.append(correction_probs[1])
                results.append(torch.stack(blocks))
            correction_probs = torch.stack(results)

            total_effect += clean_probs - corrupt_probs
            direct_effect += clean_probs - correction_probs
            indirect_effect += correction_probs - corrupt_probs
    except Exception:
        traceback.print_exc()
    finally:
        reset(activations, debug_activations, hooks, temp_hooks)

    return {
        "total": total_effect / iters,
        "direct": direct_effect / iters,
        "indirect": indirect_effect / iters,
        "clean": clean_probs,
        "corrupt": corrupt_probs,
    }
