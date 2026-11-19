"""
Input-transformation defences: mu-law compression and Butterworth filters.

Presented in PrepareMethod form.
"""
import math
from typing import Sequence, Union

import torch
from torch import Tensor
from scipy.signal import butter
from torchaudio.functional import lfilter

from .config import SAMPLE_RATE
from .prepare import PrepareFront, PrepareOverlay

#################################################
# Filters
#################################################

def pass_filter(audio: Tensor, filter_type: str, cutoff: Union[float, Sequence[float]],
                sampling_rate: int = SAMPLE_RATE, order: int = 5) -> Tensor:
    """
    Butterworth filter. ``filter_type`` is "lowpass", "highpass" (scalar cutoff) or
    "bandpass" (cutoff = [low, high]).
    """
    b, a = butter(order, cutoff, btype=filter_type, fs=sampling_rate, analog=False)
    return lfilter(audio, torch.from_numpy(a).to(audio.device).float(), torch.from_numpy(b).to(audio.device).float())

def mu_law(audio_tensor: Tensor, mu: int = 255) -> Tensor:
    """
    Mu-law compression
    """
    sign = torch.where(audio_tensor >= 0, 1, -1)
    return sign * torch.log(1 + mu * torch.abs(audio_tensor)) / math.log(1 + mu)

def inv_mu_law(audio_tensor: Tensor, mu: int = 255) -> Tensor:
    """
    Mu-law decompression
    """
    sign = torch.where(audio_tensor >= 0, 1, -1)
    return sign * (torch.tensor([mu + 1], device=audio_tensor.device).pow(torch.abs(audio_tensor)) - 1) / mu

def mu_comp_decomp(audio_tensor: Tensor, mu: int = 255) -> Tensor:
    """
    Mu-law compression followed by decompression
    """
    return inv_mu_law(mu_law(audio_tensor, mu), mu)

#################################################
# Prepare methods with a defence applied
#################################################

class PrepareFrontMuComp(PrepareFront):
    """Prepend snippet, then mu-law *compress* (no decompression)."""
    def __init__(self, snippet_size=(1, 10_240)):
        super().__init__(snippet_size)
        self.name = "prepare_front_mu_comp"

    def __call__(self, snippet, example):
        return mu_law(super().__call__(snippet, example))

class PrepareFrontMu(PrepareFront):
    """Prepend snippet, then mu-law compress + decompress."""
    def __init__(self, snippet_size=(1, 10_240)):
        super().__init__(snippet_size)
        self.name = "prepare_front_mu"

    def __call__(self, snippet, example):
        return mu_comp_decomp(super().__call__(snippet, example))

class PrepareOverlayMu(PrepareOverlay):
    """Overlay snippet, then mu-law compress."""
    def __init__(self, snippet_size=(1, 480_000)):
        super().__init__(snippet_size)
        self.name = "prepare_overlay_mu"

    def __call__(self, snippet, example):
        return mu_law(super().__call__(snippet, example))

class PrepareFrontLowpass(PrepareFront):
    def __init__(self, snippet_size=(1, 10_240), cutoff=6000):
        super().__init__(snippet_size)
        self.name = "prepare_front_lowpass"
        self.cutoff = cutoff

    def __call__(self, snippet, example):
        return pass_filter(super().__call__(snippet, example), "lowpass", cutoff=self.cutoff)

class PrepareFrontHighpass(PrepareFront):
    def __init__(self, snippet_size=(1, 10_240), cutoff=120):
        super().__init__(snippet_size)
        self.name = "prepare_front_highpass"
        self.cutoff = cutoff

    def __call__(self, snippet, example):
        return pass_filter(super().__call__(snippet, example), "highpass", cutoff=self.cutoff)

class PrepareFrontBandpass(PrepareFront):
    def __init__(self, snippet_size=(1, 10_240), cutoffs=(120, 6000)):
        super().__init__(snippet_size)
        self.name = "prepare_front_bandpass"
        self.cutoffs = list(cutoffs)

    def __call__(self, snippet, example):
        return pass_filter(super().__call__(snippet, example), "bandpass", cutoff=self.cutoffs)
