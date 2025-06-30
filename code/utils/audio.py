from IPython.display import Audio, display

import whisper

import torch
from torch import Tensor
import torchaudio

import matplotlib.pyplot as plt
from matplotlib import colormaps

from scipy.signal import butter, lfilter

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
    mel = whisper.log_mel_spectrogram(audio)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(mel, origin="lower")
    plt.show()

def mel_image(audio: torch.Tensor, pseudocolor_map: str = "Blues") -> torch.Tensor:
    image = whisper.log_mel_spectrogram(audio.squeeze())
    cm = colormaps[pseudocolor_map]
    return torch.tensor(cm(image))

def violates_clamp(snippet: torch.Tensor, clamp_epsilon: float) -> bool:
    if clamp_epsilon:
        with torch.no_grad():
            return torch.any(torch.logical_or(snippet > clamp_epsilon, snippet < -clamp_epsilon))

#################################################
# Audio filters
# - butterworth filters
# - mu law
#################################################

def lowpass_filter(audio_tensor: Tensor, cutoff: int, sampling_rate: int = 16_000, order: int = 5) -> Tensor:
    b, a = butter(order, cutoff, btype="lowpass", fs=sampling_rate, analog=False)
    y = lfilter(b, a, audio_tensor)
    return torch.from_numpy(y).to(torch.float32)

def highpass_filter(audio_tensor: Tensor, cutoff: int, sampling_rate: int = 16_000, order: int = 5) -> Tensor:
    b, a = butter(order, cutoff, btype="highpass", fs=sampling_rate, analog=False)
    y = lfilter(b, a, audio_tensor)
    return torch.from_numpy(y).to(torch.float32)

def mu_law(audio_tensor: Tensor, mu: int = 255) -> Tensor:
    sign = torch.where(audio_tensor >= 0, 1, -1)
    return sign * torch.log(1 + mu * torch.abs(audio_tensor)) / math.log(1 + mu)

def inv_mu_law(audio_tensor: Tensor, mu: int = 255) -> Tensor:
    sign = torch.where(audio_tensor >= 0, 1, -1)
    return sign * (Tensor([mu + 1]).pow(torch.abs(audio_tensor)) - 1) / mu

def mu_comp_decomp(audio_tensor: Tensor, mu: int = 255) -> Tensor:
    return inv_mu_law(mu_law(audio_tensor, mu), mu)

if __name__ == "__main__":
    print(f"{__file__} compilable!")