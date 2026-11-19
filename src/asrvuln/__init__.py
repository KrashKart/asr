"""
asrvuln: tooling for investigating the vulnerabilities of Whisper ASR models.

Submodules are imported explicitly (e.g. ``from asrvuln import attack, viz``).

    config     paths and constants
    data       TED-LIUM loading / preparation
    models     model loading (openai-whisper for attacks, HF transformers for analysis)
    audio      audio I/O and small audio helpers
    prepare    how an adversarial snippet is applied to an example (PrepareMethod)
    attack     gradient-based snippet training (complete / partial suppression)
    evaluate   attack evaluation metrics
    defences   mu-law and Butterworth filter defences
    analysis   attention, causal mediation, logit-lens and spectral analysis
    viz        plotting
    gpu        GPU memory helpers
"""

__version__ = "0.1.0"
