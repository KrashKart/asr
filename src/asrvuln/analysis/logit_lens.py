"""
"Decoder bypass" / logit lens: project every decoder block's hidden state through the output
head and read off P(EOT), to see at which depth the model decides to stop.
"""
from typing import Optional

import torch
from torch import Tensor

from .attention import inference


@torch.no_grad()
def decoder_bypass(model, processor, audio: Tensor) -> Optional[Tensor]:
    """
    Returns P(EOT) of shape (generation steps, blocks + 1); column 0 is the embedding
    output ("no blocks"). Returns None when only a single token was generated.

    Note: as in the original experiment, the final decoder layer norm is not applied
    before ``proj_out``.
    """
    eot_id = processor.tokenizer.eos_token_id
    _, res, _, _, _, _ = inference(audio, model, processor)
    # key access: inference() may return a plain dict (transformers 5.x with token timestamps)
    hidden = torch.stack([torch.stack(r) for r in res["decoder_hidden_states"][1:]]).squeeze()
    if hidden.dim() == 2:
        return None
    return torch.nn.Softmax(dim=2)(model.proj_out(hidden))[:, :, eot_id].detach().cpu()
