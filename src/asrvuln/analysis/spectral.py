"""Mel-band energy of speech vs the adversarial snippet."""
from typing import Iterable

import torch
from torch import Tensor
import whisper
from tqdm import tqdm

from ..config import N_MELS


def mel_center_freqs(n_mels: int = N_MELS, fmax: float = 8000) -> Tensor:
    """Centre frequency (Hz) of each mel band."""
    import librosa
    return torch.from_numpy(librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=fmax))


def mean_mel_energy(dataset: Iterable) -> Tensor:
    """Per-band log-mel energy, normalised by clip length, averaged over the dataset. Shape (n_mels,)."""
    total, count = torch.tensor(0.0), 0
    for d in tqdm(dataset, desc="Collating", leave=True, ncols=0):
        total = total + torch.sum(whisper.log_mel_spectrogram(d["audio"]), dim=1) / d["audio"].size(0)
        count += 1
    return total / count


def snippet_mel_energy(snippet: Tensor) -> Tensor:
    """Per-band summed log-mel energy of a snippet. Shape (n_mels,)."""
    return torch.sum(whisper.log_mel_spectrogram(snippet.squeeze().cpu()), dim=1)
