"""
Download and cache every model the repo uses, so later runs (and offline machines) don't stall.

    python scripts/download_models.py                         # tiny.en + small.en
    python scripts/download_models.py --sizes tiny.en

openai-whisper weights go to ~/.cache/whisper, HuggingFace weights to ~/.cache/huggingface.
"""
import argparse

import whisper
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", default=["tiny.en", "small.en"], help="Whisper model sizes")
    args = parser.parse_args()

    for size in args.sizes:
        print(f"openai-whisper {size} ...")
        whisper.load_model(size, device="cpu")
        print(f"HuggingFace openai/whisper-{size} ...")
        WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{size}")
        WhisperProcessor.from_pretrained(f"openai/whisper-{size}")
    print("All models cached.")


if __name__ == "__main__":
    main()
