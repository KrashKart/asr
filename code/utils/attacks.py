from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor

"""
every attack preparation method must:
1. inherit from PrepareMethod (an abstract class)
2. Override __call__

* each example can either be a batch of audio tensors (M, N) or a single audio tensor (1, N)
* the snippet must be of shape (1, N)
"""
class PrepareMethod(ABC):
    def __init__(self, snippet_size: tuple, name: str):
        assert len(snippet_size) == 2, f"Snippet must have 2 dimensions, currently has {len(snippet_size)} dims"
        assert snippet_size[0] == 1, f"Snippet must be of shape (1, N), currently of shape {snippet_size}"
        self.snippet_size = snippet_size
        self.name = name
    
    def check_dims(self, snippet: Tensor, example: Tensor, desired_dims: int = 2):
        offenders = ""
        if len(snippet.shape) != desired_dims:
            offenders += f"Need snippet (dims {len(snippet.shape)}, shape {snippet.shape}) to be of dims {desired_dims}\n"
        if len(example.shape) != desired_dims:
            offenders += f"Need example (dims {len(example.shape)}, shape {example.shape}) to be of dims {desired_dims}\n"
        if offenders:
            raise ValueError(offenders.strip())
    
    @abstractmethod
    def __call__(self, snippet: Tensor, example: Tensor):
        pass

class PrepareFront(PrepareMethod):
    def __init__(self, snippet_size=(1, 10240)):
        super().__init__(snippet_size, "prepare_front")
    
    def __call__(self, snippet, example):
        self.check_dims(snippet, example)
        snippet = snippet.repeat(example.size(0), 1)
        return torch.cat([snippet, example], dim=1)

class PrepareAtPosition(PrepareMethod):
    def __init__(self, snippet_size=(1, 10240), position=0):
        super().__init__(snippet_size, "prepare_at_position")
        self.position = position
        
    def __call__(self, snippet, example):
        self.check_dims(snippet, example)
        snippet = snippet.repeat(example.size(0), 1)
        return torch.cat([example[:, :self.position], snippet, example[:, self.position:]], dim=1)

class PrepareOverlay(PrepareMethod):
    def __init__(self, snippet_size=(1, 480_000)):
        super().__init__(snipper_size, "prepare_overlay")
    
    def __call__(self, snippet, example):
        self.check_dims(snippet, example)
        example = F.pad(example, (0, snippet.size(1) - example.size(1)), "constant", 0)
        snippet = snippet.repeat(example.size(0), 1)
        return snippet + example

class PrepareOverlayFront(PrepareMethod):
    def __init__(self, snippet_size):
        super().__init__(snippet_size, "prepare_overlay_front")
    
    def __call__(self, snippet, example):
        self.check_dims(snippet, example)
        snippet = F.pad(snippet, (0, example.size(1) - snippet.size(1)), "constant", 0)
        snippet = snippet.repeat(example.size(0), 1)
        return snippet + example
                         
class PrepareFrontMu(PrepareMethod):
    def __init__(self, snippet_size=(1, 480_000)):
        super().__init__(snippet_size, "prepare_front_mu")
    
    def __call__(self, snippet, example):
        self.check_dims(snippet, example)
        example = F.pad(example, (0, snippet.size(1) - example.size(1)), "constant", 0)
        snippet = snippet.repeat(example.size(0), 1)
        return mu_law(torch.cat([snippet, example], dim=1))
                         
class PrepareOverlayMu(PrepareMethod):
    def __init__(self, snippet_size=(1, 480_000)):
        super().__init__(snippet_size, "prepare_overlay_mu")
    
    def __call__(self, snippet, example):
        self.check_dims(snippet, example)
        example = F.pad(example, (0, snippet.size(1) - example.size(1)), "constant", 0)
        snippet = snippet.repeat(example.size(0), 1)
        return mu_law(snippet + example)

if __name__ == "__main__":
    print(f"{__file__} compilable!")