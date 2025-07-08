import torch, gc
from torch import Tensor

def print_cuda_usage(msg: str = "") -> None:
    if torch.cuda.is_available():
        print(f"{msg}{torch.cuda.memory_allocated(0)/(1024 ** 3)} GB")

def get_cuda_usage() -> Tensor:
    return torch.cuda.memory_allocated(0)/(1024 ** 3)

def clear() -> bool:
    gc.collect()
    torch.cuda.empty_cache()
    print_cuda_usage()

def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    return device