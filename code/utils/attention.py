import whisper

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Tensor

from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from tqdm import tqdm
from typing import Optional
from functools import lru_cache
from IPython.display import HTML

from . import audio

@lru_cache
def init(device: str = "cuda", size: str = "tiny.en") -> tuple:
    """
    Retrieve model and processor for the approppriate size
    """
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{size}", output_attentions=True).to(device)
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{size}")
    return model, processor

@lru_cache
def inference(audio_tensor, model, processor, skip_special_tokens=False) -> tuple:
    """
    Pass audio to model for inference. Audio tensor must be non-batched (single example) and 1-Dimensional
    """
    assert audio_tensor.dim() == 1, f"Audio tensor must be 1-Dimensional! Got {audio_tensor.dim()} dims"
    
    # Feature extraction
    inputs = processor(audio_tensor, return_tensors="pt", sampling_rate=16_000)
    input_features = inputs.input_features.to(model.device)
    
    # Generation and decoding
    res = model.generate(input_features, return_token_timestamps=True, output_attentions=True, output_hidden_states=True, return_dict_in_generate=True)
    decoded = processor.decode(res.sequences.squeeze(), skip_special_tokens=skip_special_tokens)
    
    # Token timestamps
    list_of_tokens = [processor.decode(r, skip_special_tokens=skip_special_tokens) for r in res.sequences.squeeze()]
    timestamps = list(zip(list_of_tokens, res.token_timestamps.squeeze().tolist()))
    
    return (decoded, res, timestamps, res.encoder_attentions, res.decoder_attentions, res.cross_attentions)

@lru_cache
def plot_attns(attns: Tensor, rows: int, cols: int, 
               figsize: Optional[tuple] = None, cmap: str = "viridis",
               filename: Optional[str] = None) -> None:
    """
    Plot the attention maps in a grid
    """
    attns = [a.cpu().squeeze(0) for a in attns]
    
    blocks = len(attns)
    heads = attns[0].size(0)
    assert rows * cols == blocks * heads
    
    if not figsize:
        figsize = (cols * 10, rows * 10)
    
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    pbar = tqdm(range(blocks * heads), leave=True, ncols=0)
    
    for idx in pbar:
        i, j = idx // cols, idx % cols
        x, y = idx // heads, idx % heads
        sns.heatmap(attns[x][y,:,:], cmap="viridis", ax=ax[i][j])
        ax[i][j].set_title(f"Block {x + 1} Head {y + 1}")
        pbar.refresh()

    if filename:
        plt.savefig(filename)
    plt.show()

@lru_cache
def plot_attns_over_iters(attns: Tensor, rows: int, cols: int,
                          vmin: float = 0.0, vmax: float = 1.0,
                          figsize: Optional[tuple] = None, cmap: str = "viridis",
                          filename: Optional[str] = None) -> None:
    """
    Plot the attention maps in a grid
    """
    assert rows * cols == attns.size(0)
    attns_list = [a.cpu().squeeze(0) for a in attns]
    
    if not figsize:
        figsize = (cols * 10, rows * 10)
    
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    pbar = tqdm(range(attns.size(0)), leave=True, ncols=0)
    
    for idx in pbar:
        i, j = idx // cols, idx % cols
        sns.heatmap(attns_list[idx], cmap="viridis", ax=ax[i][j], vmin=vmin, vmax=vmax)
        ax[i][j].set_title(f"Iter {idx + 1}")
        pbar.refresh()

    if filename:
        plt.savefig(filename)
    plt.show()
    
@lru_cache
def smoothen(frames, factor=5):
    result = []
    for i in range(frames.size(0)):
        if i != 0:
            step_diff = (frames[i, :, :] - frames[i - 1, :, :]) / (factor - 1)
            for j in range(factor - 1):
                result.append(frames[i - 1, :, :] + j * step_diff)
        result.append(frames[i, :, :])
    assert len(result) == (frames.size(0) - 1) * factor + 1
    return torch.stack(result)

@lru_cache
def plot_attns_iters_anim(attns,
                          vmin: float = 0.0, vmax: float = 1.0,
                          figsize: tuple = (5, 5), cmap: str = "viridis",
                          interval: int = 500) -> None:
    """
    Plot the attention maps over iterations as animation
    """
    attns = attns.squeeze().cpu()
    fig, ax = plt.subplots(figsize=figsize)
    hm = sns.heatmap(attns[0,:,:], cmap=cmap, vmin=vmin, vmax=vmax, ax=ax)

    def update(frame):
        # ax.collections[-1].colorbar.remove()
        hm.collections[0].set_array(attns[frame, :, :])
        return hm

    anim = animation.FuncAnimation(fig=fig, func=update, frames=attns.size(0), interval=interval, repeat=False)
    return HTML(anim.to_jshtml()), anim

@lru_cache
def get_spikes(attn: Tensor, lim: float) -> tuple:
    """
    Return the indices and timestamps that cause spikes in attention
    """
    maxi, mini = attn.max(), attn.min()
    normalised_attn = (attn - mini) / (maxi - mini)
    masked = torch.where(normalised_attn >= lim, 1.0, 0.0)
    indices = masked.nonzero()[:, 1].unique().cpu()
    seconds = indices * 0.02 * 16_000
    return indices, seconds

@lru_cache
def plot_spikes(audio: Tensor, attn: Tensor, lim: float, figsize: tuple = (20, 6), filename: str = None, draw_on_mel: bool = True) -> None:
    """
    PLot the attention map, waveform and log mel spectrograms and indicate attention spikes on waveform
    """
    fig, ax = plt.subplots(2, 2, figsize=figsize)
    gs = ax[1, 0].get_gridspec()
    ax[1, 0].remove()
    ax[1, 1].remove()
    
    sns.heatmap(attn.detach().cpu(), cmap="viridis", ax=ax[0, 0])
    ax[0, 0].set_title("Attention Map")
    ax[0, 0].set_xlabel("Encoder tokens (0.02s each)")
    
    indices, seconds = get_spikes(attn, lim)
    print(f"Indices: {indices.tolist()}")
    
    half_second_frames = torch.arange(0, audio.size(0), 8_000)
    half_seconds = half_second_frames / 16_000
    half_seconds = half_seconds.tolist()

    ax[0, 1].plot(audio)
    ax[0, 1].vlines(seconds, -1, 1, color="r")
    ax[0, 1].set_title("Waveform")
    ax[0, 1].set_xticks(half_second_frames, half_seconds)
    ax[0, 1].set_xlabel("Seconds (s)")
    
    bigax = fig.add_subplot(gs[1, :])
    bigax.imshow(whisper.log_mel_spectrogram(audio), origin="lower")
    if draw_on_mel:
        bigax.vlines(seconds / 16_000 * 100, 0, 79, color="r")
    bigax.set_title("Log Mel Spectrogram")
    bigax.set_xlabel("Seconds (ms)")
    
    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    
    if filename:
        plt.savefig(filename)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    print(f"{__file__} compilable!")