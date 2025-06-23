from IPython.display import Audio, display

import torch
import torchaudio

def save_audio(audio_tensor: torch.Tensor, filename: str = "test.wav", sample_rate: int = 16_000) -> None:
    """
    Saves audio tensor as a wav file
    """
    if len(audio_tensor.shape) == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    elif audio_tensor.size(0) != 1:
        raise ValueError(f"audio_tensor must be of dims (1, n), current dims are {audio_tensor.shape}")
    torchaudio.save(filename, audio_tensor.detach().to("cpu"), sample_rate)

def load_audio(filename: str = "test.wav") -> torch.Tensor:
    """
    Loads audio from wav
    """
    return torchaudio.load(filename)[0]

def play_audio(param = "test.wav", sample_rate: int = 16_000):
    """
    Plays audio tensor or audio file
    """
    if isinstance(param, torch.Tensor):
        display(Audio(param, rate=sample_rate))
    else:
        display(Audio(param))