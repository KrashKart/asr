"""GPU memory management utils."""

import gc
from typing import List

import torch
from torch import Tensor

def print_cuda_usage(msg: str = "") -> None:
    if torch.cuda.is_available():
        print(f"{msg}{torch.cuda.memory_allocated(0)/(1024 ** 3)} GB")

def get_cuda_usage() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(0)/(1024 ** 3)

def get_total_memory(*tensors: Tensor) -> List[int]:
    return [t.element_size() * t.nelement() for t in tensors]

def cleanup(verbose: bool = False) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    if verbose:
        print_cuda_usage("GPU Cleared: ")

def clear() -> None:
    cleanup()
    print_cuda_usage()

def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    return device
