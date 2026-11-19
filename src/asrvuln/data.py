"""
TED-LIUM loading and preparation.

Some functions made with Claude.

On-disk layout (under ``config.TEDLIUM_DIR``):

train.hf  validation.hf  test.hf            # columns: audio, text
train_idx.hf  validation_idx.hf             # + idx column (used for success tracking)
"""
import io
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from datasets import Dataset, load_dataset, load_from_disk

from . import config

_SPLITS = {"train": "train", "validation": "validation", "valid": "validation", "test": "test"}

def tedlium_path(split: str, with_idx: bool = False, root: Optional[Union[str, Path]] = None) -> Path:
    if split not in _SPLITS:
        raise ValueError(f"Unknown split {split!r}, expected one of {sorted(_SPLITS)}")
    root = Path(root) if root else config.TEDLIUM_DIR
    return root / f"{_SPLITS[split]}{'_idx' if with_idx else ''}.hf"

def load_tedlium(split: str, n: Optional[int] = None, seed: Optional[int] = None,
                 with_idx: bool = False, root: Optional[Union[str, Path]] = None) -> Dataset:
    """
    Load TEDLIUM (release 3).

    split    : "train", "validation" (or "valid") or "test"
    n        : keep only the first n examples (after shuffling, if seed is given)
    seed     : seed for shuffling examples
    with_idx : load with index column
    """
    ds = load_from_disk(str(tedlium_path(split, with_idx, root))).with_format("torch")
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    if n is not None:
        ds = ds.select(range(n))
    return ds

def collate_idx(batch):
    """
    DataLoader collate_fn returning ``(audio, idx)``. Used by ``attack.train``.

    batch_size = 1 : audio of shape (1, N), idx an int.
    batch_size > 1 : audio right-zero-padded to the longest clip, shape (B, N_max), idx a list of ints.
                   Whisper zero-pads to 30 s anyway, so the padding doesn't change the mel input.
    """
    if len(batch) == 1:
        return batch[0]["audio"].unsqueeze(0), batch[0]["idx"].item()
    audio = torch.nn.utils.rnn.pad_sequence([b["audio"] for b in batch], batch_first=True)
    return audio, [b["idx"].item() for b in batch]

def collate_audio(batch):
    """DataLoader collate_fn (batch_size = 1) returning only the audio, shape (1, N)."""
    return batch[0]["audio"].unsqueeze(0)

# TED-LIUM release 3 as plain parquet
TEDLIUM_REPO = "sanchit-gandhi/tedlium-data"

def _tedlium_shards(split: str, n: Optional[int] = None, cache_dir: Optional[str] = None) -> List[str]:
    """Download (and cache) the parquet shards of a split, stopping once ``n`` rows are covered."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    names = sorted(f for f in HfApi().list_repo_files(TEDLIUM_REPO, repo_type="dataset")
                   if f.startswith(f"data/{split}-") and f.endswith(".parquet"))
    if not names:
        raise FileNotFoundError(f"No parquet shards for split {split!r} in {TEDLIUM_REPO}")

    paths, rows = [], 0
    for name in names:
        paths.append(hf_hub_download(TEDLIUM_REPO, name, repo_type="dataset", cache_dir=cache_dir))
        rows += pq.ParquetFile(paths[-1]).metadata.num_rows
        if n is not None and rows >= n:
            break
    return paths

def _decode_wav(audio: dict) -> np.ndarray:
    """WAV bytes -> mono float32 at ``config.SAMPLE_RATE`` (soundfile: no torchcodec/FFmpeg needed)."""
    import soundfile as sf

    samples, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        import librosa
        samples = librosa.resample(samples, orig_sr=sr, target_sr=config.SAMPLE_RATE)
    return samples

def _keep_audio_and_text(x: dict) -> dict:
    return {"audio": _decode_wav(x["audio"]), "text": x["text"]}

def prepare_tedlium(root: Optional[Union[str, Path]] = None, n_train: int = 2_000,
                    cache_dir: Optional[str] = None) -> None:
    """
    Download TED-LIUM (release 3) and save the splits used in this project (see module docstring)
    to ``root`` (default ``config.TEDLIUM_DIR``).

    Only the first ``n_train`` training examples are kept, and only the shards needed for them are
    downloaded; validation and test are kept in full. Each example keeps just the decoded audio
    (float32, 16 kHz) and its ground-truth transcript ``text``.
    """
    from datasets import Audio

    root = Path(root) if root else config.TEDLIUM_DIR
    root.mkdir(parents=True, exist_ok=True)

    drop = ["speaker_id", "gender", "file", "id"]

    splits = {}
    for name, n in (("train", n_train), ("validation", None), ("test", None)):
        ds = load_dataset("parquet", data_files=_tedlium_shards(name, n, cache_dir), split="train", cache_dir=cache_dir)
        ds = ds.cast_column("audio", Audio(decode=False))
        if n is not None:
            ds = ds.select(range(min(n, len(ds))))
        splits[name] = ds.map(_keep_audio_and_text, remove_columns=drop, desc=f"Decoding {name}")
        print(f"{name}: {len(splits[name])} examples")
        splits[name].save_to_disk(str(tedlium_path(name, root=root)))

    for name in ("train", "validation"):
        ds = splits[name]
        ds.add_column("idx", range(len(ds))).save_to_disk(str(tedlium_path(name, with_idx=True, root=root)))
