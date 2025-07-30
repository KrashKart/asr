from IPython.display import Audio, display
import math

import whisper

import torch
from torch import Tensor
import torchaudio

import matplotlib.pyplot as plt
from matplotlib import colormaps

from scipy.signal import butter
from torchaudio.functional import lfilter

def save_audio(audio_tensor: Tensor, filename: str = "test.wav", sample_rate: int = 16_000) -> None:
    """
    Saves audio tensor as a wav file
    """
    if len(audio_tensor.shape) == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    elif audio_tensor.size(0) != 1:
        raise ValueError(f"audio_tensor must be of dims (1, n), current dims are {audio_tensor.shape}")
    torchaudio.save(filename, audio_tensor.detach().to("cpu"), sample_rate)

def load_audio(filename: str = "test.wav") -> Tensor:
    """
    Loads audio from wav
    """
    return torchaudio.load(filename)[0]

def play_audio(param = "test.wav", sample_rate: int = 16_000):
    """
    Plays audio tensor or audio file
    """
    if isinstance(param, Tensor):
        display(Audio(param, rate=sample_rate))
    else:
        display(Audio(param))

def view_mel(audio: Tensor, figsize: tuple = (20, 6)):
    """
    View the log mel spectrogram in matplotlib
    """
    mel = whisper.log_mel_spectrogram(audio)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(mel, origin="lower")
    plt.show()

def mel_image(audio: torch.Tensor, pseudocolor_map: str = "Blues") -> torch.Tensor:
    """
    Get the image tensor for the log mel spectrogram
    
    Deprecated since the color schemes damn ugly
    """
    image = whisper.log_mel_spectrogram(audio.squeeze())
    cm = colormaps[pseudocolor_map]
    return torch.tensor(cm(image))

def violates_clamp(snippet: torch.Tensor, clamp_epsilon: float) -> bool:
    """
    Checks if the snippet violates the clamp limit constraint
    """
    if clamp_epsilon:
        with torch.no_grad():
            return torch.any(torch.logical_or(snippet > clamp_epsilon, snippet < -clamp_epsilon))

#################################################
# Audio filters
# - butterworth filters
# - mu law
#################################################

def pass_filter(audio: Tensor, filter_type, cutoff: int, sampling_rate: int = 16_000, order: int = 5) -> Tensor:
    """
    Toggleable low-/high-pass butterworth filter
    """
    b, a = butter(order, cutoff, btype=filter_type, fs=sampling_rate, analog=False)
    y = lfilter(audio, torch.from_numpy(a).to(audio.device).float(), torch.from_numpy(b).to(audio.device).float())
    return y

def mu_law(audio_tensor: Tensor, mu: int = 255) -> Tensor:
    """
    Mu Law compression
    """
    sign = torch.where(audio_tensor >= 0, 1, -1)
    return sign * torch.log(1 + mu * torch.abs(audio_tensor)) / math.log(1 + mu)

def inv_mu_law(audio_tensor: Tensor, mu: int = 255) -> Tensor:
    """
    Mu Law decompression
    """
    sign = torch.where(audio_tensor >= 0, 1, -1)
    return sign * (torch.tensor([mu + 1], device=audio_tensor.device).pow(torch.abs(audio_tensor)) - 1) / mu

def mu_comp_decomp(audio_tensor: Tensor, mu: int = 255) -> Tensor:
    """
    duh
    """
    return inv_mu_law(mu_law(audio_tensor, mu), mu)
