from typing import Union, List, Dict, Callable, Any
from transformers import WhisperForConditionalGeneration

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle

import traceback
import time

IntTensorDict = Dict[int, List[Tensor]]
Number = Union[int, float]
Whisper = WhisperForConditionalGeneration
Hook = Callable[..., Any]

def reset(*args: Union[List[RemovableHandle], IntTensorDict]) -> None:
    for l in args:
        if isinstance(l, list):
            for handle in l:
                handle.remove()
        l.clear()
        assert not l, f"{l} cannot be cleared!"

def append_dict(d: IntTensorDict, key: int, value: List[Tensor]) -> None:
    temp = d.get(key, [])
    temp.append(value)
    d[key] = temp

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
        assert output.shape == (output + epsilons).shape, "ERROR"
        return output + epsilons
    return hook

def correction_hook(block: int, token: int, activations: IntTensorDict):
    activation_count = 0
    def hook(module, args, out):
        nonlocal activation_count
        activation_count += 1
        if activation_count == token:
            out = activations[block][token]
        return out
    return hook

#####################################
# Registration
#####################################
def register_all_decoder_blocks(model: Whisper, hook_creator: Hook, activations: IntTensorDict, hl: List[RemovableHandle]) -> None:
    for i, block in enumerate(model.model.decoder.layers):
        hook_i = hook_creator(i, activations)
        h = block.register_forward_hook(hook_i)
        hl.append(h)
        
def register_at_decoder_block(model: Whisper, hook_creator: Hook, block: int, hl: List[RemovableHandle], *args: Any) -> RemovableHandle:
    h = model.model.decoder.layers[block].register_forward_hook(hook_creator(*args))
    hl.append(h)
    return h

def register_correction_hook(model: Whisper, block: int, token: int, hl: List[RemovableHandle]) -> RemovableHandle:
    h = model.model.decoder.layers[block].register_forward_hook(correction_hook(block, token))
    hl.append(h)
    return h

def register_encoder_embedding_hook(model: Whisper, hook_creator: Hook, hl: List[RemovableHandle], alpha: int = 1) -> RemovableHandle:
    h = model.model.encoder.embed_tokens.register_forward_hook(hook_creator(alpha))
    hl.append(h)
    return h

def register_decoder_embedding_hook(model: Whisper, hook_creator: Hook, hl: List[RemovableHandle], alpha: int = 1) -> RemovableHandle:
    h = model.model.decoder.embed_tokens.register_forward_hook(hook_creator(alpha))
    hl.append(h)
    return h
    