"""
Configs for this repo.

Paths default to ``/data`` and ``/outputs``
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"

TEDLIUM_DIR = DATA_DIR / "tedlium"
SNIPPETS_DIR = OUTPUT_DIR / "snippets"   # trained snippets (.pt)
TENSORS_DIR = OUTPUT_DIR / "tensors"     # other tensors
RUNS_DIR = OUTPUT_DIR / "runs"           # tensorboard logs
FIGURES_DIR = OUTPUT_DIR / "figures"

SAMPLE_RATE = 16_000
ENCODER_FPS = 50      # Whisper encoder tokens per second (20 ms each)
MEL_FPS = 100         # log-mel frames per second (10 ms hop)
N_MELS = 80

DEFAULT_SNIPPET_SIZE = (1, 10_240)
DEFAULT_CLAMP_EP = 0.005
