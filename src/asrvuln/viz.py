"""Plotting utils.
    
Functions take tensors and draw; pass ``filename`` to also save the figure.
Made with Claude.
"""
from typing import Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor
import whisper
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from IPython.display import HTML
from tqdm import tqdm

from .config import MEL_FPS, SAMPLE_RATE
from .analysis.attention import get_spikes

def _finish(filename: Optional[str]) -> None:
    """Save fig and show fig."""
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()

#################################################
# Audio
#################################################

def view_mel(audio: Tensor, figsize: tuple = (20, 6), filename: Optional[str] = None) -> None:
    """Log mel spectrogram of one clip."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(whisper.log_mel_spectrogram(audio.squeeze().cpu()), origin="lower")
    _finish(filename)

def view_mels(examples: Sequence[Mapping], figsize: Optional[tuple] = None, filename: Optional[str] = None) -> None:
    """Stacked log mel spectrograms of dataset items (needs "audio" and "idx" keys)."""
    figsize = figsize or (10, len(examples) * 1.5)
    fig, ax = plt.subplots(len(examples), 1, figsize=figsize, squeeze=False)
    for i, ex in enumerate(examples):
        ax[i, 0].imshow(whisper.log_mel_spectrogram(ex["audio"]))
        ax[i, 0].set_title(f"Example {ex['idx']}")
    plt.tight_layout()
    _finish(filename)

def plot_waveform_and_mel(audio: Tensor, snippet_len: Optional[int] = None,
                          figsize: tuple = (40, 5), filename: Optional[str] = None) -> None:
    """Waveform and log mel side by side, with a dashed line where the snippet ends."""
    audio = audio.squeeze().cpu()
    seconds = audio.size(0) / SAMPLE_RATE
    mel = whisper.log_mel_spectrogram(audio)

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].plot(audio)
    ax[0].set_title("Audio Waveform")
    ticks = torch.arange(0, audio.size(0) + 1, SAMPLE_RATE // 2)
    ax[0].set_xticks(ticks, [f"{t / SAMPLE_RATE:g}" for t in ticks.tolist()])
    ax[0].set_xlim(0, audio.size(0))
    ax[0].set_xlabel("Seconds (s)")

    ax[1].imshow(mel, origin="lower", aspect="auto")
    ax[1].set_title("Log Mel Spectrogram")
    mel_ticks = torch.arange(0, int(seconds * MEL_FPS) + 1, MEL_FPS // 2)
    ax[1].set_xticks(mel_ticks, [f"{t / MEL_FPS:g}" for t in mel_ticks.tolist()])
    ax[1].set_xlim(0, seconds * MEL_FPS)
    ax[1].set_xlabel("Seconds (s)")

    if snippet_len:
        ax[0].axvline(x=snippet_len, color="r", linestyle="--")
        ax[1].axvline(x=snippet_len / SAMPLE_RATE * MEL_FPS, color="r", linestyle="--")
    plt.tight_layout()
    _finish(filename)

def plot_mel_band_energy(energies: Dict[str, Tensor], figsize: tuple = (16, 4), filename: Optional[str] = None) -> None:
    """One bar chart per entry of per-mel-band energy (see analysis.spectral)."""
    fig, ax = plt.subplots(1, len(energies), figsize=figsize, squeeze=False)
    for a, (name, e) in zip(ax[0], energies.items()):
        a.bar(x=range(e.size(0)), height=e)
        a.set_title(name)
        a.set_xlabel("Mel band")
    plt.tight_layout()
    _finish(filename)

#################################################
# Attention
#################################################

def plot_attns(attns: Tensor, rows: int, cols: int,
               figsize: Optional[tuple] = None, cmap: str = "viridis",
               filename: Optional[str] = None) -> None:
    """
    Plot (blocks, heads, X, Y) attention maps in a rows x cols grid.
    rows * cols must equal blocks * heads (tiny.en: 4 * 6 = 24).
    """
    attns = attns.cpu().squeeze()
    blocks, heads = attns.size(0), attns.size(1)
    assert rows * cols == blocks * heads, f"rows * cols ({rows * cols}) != blocks * heads ({blocks * heads})"

    figsize = figsize or (cols * 10, rows * 10)
    fig, ax = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    for idx in tqdm(range(blocks * heads), leave=True, ncols=0):
        i, j = idx // cols, idx % cols
        x, y = idx // heads, idx % heads
        sns.heatmap(attns[x][y, :, :], cmap=cmap, ax=ax[i][j])
        ax[i][j].set_title(f"Block {x + 1} Head {y + 1}")
    _finish(filename)

def plot_attns_over_iters(attns: Tensor, rows: int, cols: int,
                          vmin: float = 0.0, vmax: float = 1.0,
                          figsize: Optional[tuple] = None, cmap: str = "viridis",
                          filename: Optional[str] = None) -> None:
    """Plot (iters, X, Y) attention maps of one head in a rows x cols grid (rows * cols == iters)."""
    assert rows * cols == attns.size(0)
    attns = attns.cpu().squeeze()
    figsize = figsize or (cols * 10, rows * 10)
    fig, ax = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    for idx in tqdm(range(attns.size(0)), leave=True, ncols=0):
        i, j = idx // cols, idx % cols
        sns.heatmap(attns[idx, ...], cmap=cmap, ax=ax[i][j], vmin=vmin, vmax=vmax)
        ax[i][j].set_title(f"Iter {idx + 1}")
    _finish(filename)

def plot_attns_iters_anim(attns: Tensor, vmin: float = 0.0, vmax: float = 1.0,
                          figsize: tuple = (5, 5), cmap: str = "viridis", interval: int = 500) -> tuple:
    """
    Animate (frames, X, Y) attention maps. Returns (HTML for notebook display, FuncAnimation);
    save with ``anim.save("file.mp4")``.
    """
    attns = attns.squeeze().cpu()
    fig, ax = plt.subplots(figsize=figsize)
    hm = sns.heatmap(attns[0, :, :], cmap=cmap, vmin=vmin, vmax=vmax, ax=ax)

    def update(frame):
        hm.collections[0].set_array(attns[frame, :, :])
        return hm

    fig.tight_layout()
    anim = animation.FuncAnimation(fig=fig, func=update, frames=attns.size(0), interval=interval, repeat=False)
    plt.close(fig)
    return HTML(anim.to_jshtml()), anim

def plot_spikes(audio: Tensor, attn: Tensor, lim: float, cmap: str = "viridis", figsize: tuple = (20, 6),
                filename: Optional[str] = None, draw_on_mel: bool = True) -> None:
    """
    Plot an attention map, the waveform and the log mel spectrogram, marking the audio
    positions whose attention spikes above ``lim`` (normalised).
    """
    audio = audio.detach().cpu()
    fig, ax = plt.subplots(2, 2, figsize=figsize)
    gs = ax[1, 0].get_gridspec()
    ax[1, 0].remove()
    ax[1, 1].remove()

    sns.heatmap(attn.detach().cpu(), cmap=cmap, ax=ax[0, 0])
    ax[0, 0].set_title("Attention Map")
    ax[0, 0].set_xlabel("Encoder tokens (0.02s each)")

    indices, samples = get_spikes(attn, lim)
    print(f"Indices: {indices.tolist()}")

    half_second_frames = torch.arange(0, audio.size(0), SAMPLE_RATE // 2)
    ax[0, 1].plot(audio)
    ax[0, 1].vlines(samples, -1, 1, color="r")
    ax[0, 1].set_title("Waveform")
    ax[0, 1].set_xticks(half_second_frames, (half_second_frames / SAMPLE_RATE).tolist())
    ax[0, 1].set_xlabel("Seconds (s)")

    bigax = fig.add_subplot(gs[1, :])
    bigax.imshow(whisper.log_mel_spectrogram(audio), origin="lower")
    if draw_on_mel:
        bigax.vlines(samples / SAMPLE_RATE * MEL_FPS, 0, 79, color="r")
    bigax.set_title("Log Mel Spectrogram")
    bigax.set_xlabel("Mel frames (10 ms each)")

    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    plt.tight_layout()
    _finish(filename)

def plot_attn_head(attn: Tensor, title: str = "", figsize: tuple = (10, 10), cmap: str = "viridis",
                   filename: Optional[str] = None) -> None:
    """Single cross-attention head heatmap (decoder tokens x encoder tokens)."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(attn.detach().cpu(), ax=ax, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Encoder Tokens")
    ax.set_ylabel("Decoder Tokens")
    _finish(filename)

#################################################
# Attack / analysis results
#################################################

def plot_success_heatmap(success: Tensor, ids: Optional[Sequence[int]] = None,
                         figsize: tuple = (10, 15), filename: Optional[str] = None) -> None:
    """Transcript length per example (rows) per training iteration (columns)."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(success, cmap="magma", ax=ax, yticklabels=list(ids) if ids is not None else False)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Example")
    _finish(filename)

def plot_eos_by_layer(probs: Optional[Tensor], title: str = "", filename: Optional[str] = None) -> None:
    """
    Plot ``analysis.logit_lens.decoder_bypass`` output (steps, blocks + 1): log P(EOT) per
    block at the final step, and per block across generation steps.
    """
    if probs is None:
        print(f"{title}: only one token generated, nothing to plot")
        return
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    n_layers = probs.size(1)
    ax[0].plot(torch.log10(probs[-1]), marker="o")
    ax[0].set_xticks(range(n_layers), ["emb"] + [f"B{i}" for i in range(1, n_layers)])
    ax[0].set_xlabel("Decoder block")
    ax[0].set_ylabel("log10 P(EOT)")
    ax[0].set_title(f"{title} (final step)")
    for layer in range(n_layers):
        ax[1].plot(torch.log10(probs[:, layer]), label="Embedding" if layer == 0 else f"Block {layer}")
    ax[1].set_xlabel("Generation step")
    ax[1].set_ylabel("log10 P(EOT)")
    ax[1].legend()
    plt.tight_layout()
    _finish(filename)

def plot_effects(effects: Dict[str, Tensor], filename: Optional[str] = None) -> None:
    """Heatmaps of ``analysis.causal.run_causal_trace`` output: |IE|, TE, DE and |IE| / |TE|."""
    panels = {
        "|IE| = |P(eot | corrected) - P(eot | corrupted)|": torch.abs(effects["indirect"]),
        "TE = P(eot | clean) - P(eot | corrupted)": effects["total"],
        "DE = P(eot | clean) - P(eot | corrected)": effects["direct"],
        "|IE| / |TE|": torch.abs(effects["indirect"]) / torch.abs(effects["total"]),
    }
    fig, ax = plt.subplots(1, 4, figsize=(24, 5))
    for a, (title, value) in zip(ax, panels.items()):
        kwargs = {"vmin": 0.0, "vmax": 2.0} if title.startswith("|IE| /") else {}
        sns.heatmap(value, ax=a, **kwargs)
        a.set_title(title)
        a.set_xlabel("Restored block")
        a.set_ylabel("Token restored at")
        a.invert_yaxis()
    plt.tight_layout()
    _finish(filename)
