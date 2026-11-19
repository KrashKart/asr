"""
Model loading.

Two Whisper implementations used:
  * openai-whisper (``load_whisper``) for training/evaluating attacks due to differentiable ``model.forward(mel, tokens)`` and ``model.transcribe``.
  * HuggingFace transformers (``load_hf_whisper``) for analysis because ``generate`` can return attentions, hidden states and token timestamps.
"""
from typing import Optional, Tuple

import whisper
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .gpu import get_device

__all__ = ["get_device", "load_whisper", "load_hf_whisper"]

def load_whisper(name: str = "tiny.en", device: Optional[str] = None) -> whisper.model.Whisper:
    """Load openai-whisper model in eval mode."""
    device = device or get_device()
    model = whisper.load_model(name).to(device)
    model.eval()
    return model

def load_hf_whisper(size: str = "tiny.en", device: Optional[str] = None,
                    output_attentions: bool = True,
                    local_files_only: bool = False) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """
    Load HuggingFace Whisper model (eval mode) and its processor.

    Each call is fresh for CMA to hold 2 copies.
    Weights are cached by HuggingFace after the first download.
    Can pass ``local_files_only=True`` if needed.
    """
    device = device or get_device()
    model = WhisperForConditionalGeneration.from_pretrained(
        f"openai/whisper-{size}", output_attentions=output_attentions, local_files_only=local_files_only
    ).to(device)
    model.eval()
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{size}", local_files_only=local_files_only)
    return model, processor
