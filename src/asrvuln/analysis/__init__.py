"""
Analysis of *why* the attack works (HuggingFace Whisper models, see ``models.load_hf_whisper``).

    attention   inference with attentions, attention stacking helpers, attention spikes
    causal      causal mediation analysis (corrupt / restore decoder blocks)
    logit_lens  P(EOT) read out at every decoder block ("decoder bypass")
    spectral    mel-band energy of speech vs the adversarial snippet
"""
