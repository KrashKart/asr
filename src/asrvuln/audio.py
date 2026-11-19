"""Audio utils"""
import soundfile as sf
import torch
from torch import Tensor
import whisper
from matplotlib import colormaps
from IPython.display import Audio, display

from .config import SAMPLE_RATE

def save_audio(audio_tensor: Tensor, filename: str = "test.wav", sample_rate: int = SAMPLE_RATE) -> None:
    """
    Saves audio tensor as a wav file
    """
    if len(audio_tensor.shape) == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    elif audio_tensor.size(0) != 1:
        raise ValueError(f"audio_tensor must be of dims (1, n), current dims are {audio_tensor.shape}")
    sf.write(filename, audio_tensor.detach().cpu().squeeze(0).float().numpy(), sample_rate)

def load_audio(filename: str = "test.wav") -> Tensor:
    """
    Loads audio from wav as a float32 tensor of shape (channels, N)
    """
    samples, _ = sf.read(filename, dtype="float32", always_2d=True)  # (N, channels)
    return torch.from_numpy(samples.T.copy())

def play_audio(param="test.wav", sample_rate: int = SAMPLE_RATE):
    """
    Plays audio tensor or audio file (in a notebook)
    """
    if isinstance(param, Tensor):
        display(Audio(param.detach().cpu(), rate=sample_rate))
    else:
        display(Audio(param))

def mel_image(audio: Tensor, pseudocolor_map: str = "Blues") -> Tensor:
    """
    Get an RGBA image tensor (H, W, 4) of the log mel spectrogram, e.g. for tensorboard
    """
    image = whisper.log_mel_spectrogram(audio.squeeze())
    cm = colormaps[pseudocolor_map]
    return torch.tensor(cm(image))

def violates_clamp(snippet: Tensor, clamp_epsilon: float) -> bool:
    """
    Checks if the snippet violates the clamp limit constraint
    """
    if not clamp_epsilon:
        return False
    with torch.no_grad():
        return bool(torch.any(torch.logical_or(snippet > clamp_epsilon, snippet < -clamp_epsilon)))

def mean_amplitude_filler(snippet: Tensor, example: Tensor) -> Tensor:
    """
    Fill a similar snippet with mean amplitude for testing.
    """
    return torch.full(snippet.shape, torch.mean(torch.abs(example)).item())
