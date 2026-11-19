# Investigating the Vulnerabilities of Automatic Speech Recognition (ASR) Models
This is the repo used for my DSO Internship Project (named titularly), also under the National University of Singapore's Student Internship Programme (CP3200: SIP I).

The preprint on arXiv can be found [here](https://arxiv.org/abs/2508.09994)!

## What's in here
The project studies OpenAI Whisper on TED-LIUM in three parts:

* **Attack**: a short universal adversarial snippet (10,240 samples / 0.64 s, L∞ ≤ 0.005) is prepended to speech so Whisper emits end-of-transcript straight away.
  * *Complete suppression*: maximise P(EOT) at the first token, replicated from [prepend_acoustic_attack](https://github.com/rainavyas/prepend_acoustic_attack).
  * *Partial suppression*: maximise P(EOT) over the first few tokens.
* **Model analysis**: why the attack works.
  * Encoder, decoder and cross attention maps.
  * Decoder bypass (logit lens).
  * Causal mediation analysis, inspired by [ROME](https://arxiv.org/abs/2202.05262).
  * A differential comparison against a mean-amplitude filler.
  * The snippet's frequency profile.
* **Defences**: mu-law compression / decompression and Butterworth low-, high- and band-pass filters.

## Repo structure
```
src/asrvuln/          the library: all functionality lives here
  config.py           paths (data/, outputs/) and constants
  data.py             TED-LIUM loading / preparation
  models.py           openai-whisper (attacks) and HF Whisper (analysis) loaders
  audio.py            audio I/O helpers
  prepare.py          how a snippet is applied to audio (PrepareFront, PrepareAtPosition, ...)
  attack.py           snippet training: forward (complete), forward_multi (partial), train
  evaluate.py         empty rate, ASL, BLEU, WER, SNR, STOI
  defences.py         mu-law, Butterworth filters and their PrepareMethod variants
  analysis/           attention, causal, logit_lens, spectral
  viz.py              all plotting
  gpu.py              GPU memory helpers
notebooks/
  01_attack_and_defence.ipynb   demo: load/train a snippet, evaluate it, test defences
  02_model_analysis.ipynb       demo: attention, decoder bypass, causal mediation, frequency
archive/              original exploratory notebooks (kept for their outputs)
scripts/
  prepare_data.py     download TED-LIUM -> data/tedlium/
  train_attack.py     train a snippet -> outputs/snippets/<mode>_snippets.pt
  evaluate_attack.py  evaluate a snippet, optionally against a defence
tests/                CPU smoke tests
data/, outputs/       datasets and generated artefacts (git-ignored)
```

## From scratch
Run these in order from the repo root, inside a fresh venv:
```bash
pip install -r requirements.txt          # 1. dependencies + this repo (editable)
pytest                                   # 2. sanity check, CPU only, a few seconds
python scripts/download_models.py        # 3. cache Whisper tiny.en + small.en (openai-whisper and HuggingFace)
python scripts/prepare_data.py           # 4. TED-LIUM -> data/tedlium/
python scripts/train_attack.py --mode complete   # 5. snippet -> outputs/snippets/complete_snippets.pt (~45 min on a GPU)
python scripts/train_attack.py --mode partial --gap 0.01 --train-n 250 --valid-n 100 --mins-limit 0   # optional
```
Then open the notebooks in `notebooks/` (pick the venv as the Jupyter kernel).
* Demo 01 runs even without step 5: it trains a quick, weak snippet instead.
* Demo 02 needs step 5.

To keep data elsewhere, set `ASRVULN_DATA_DIR` and/or `ASRVULN_OUTPUT_DIR`.

## Other commands
```bash
python scripts/evaluate_attack.py --snippet outputs/snippets/complete_snippets.pt
python scripts/evaluate_attack.py --snippet outputs/snippets/complete_snippets.pt --defence lowpass --cutoff 7500
python scripts/evaluate_attack.py --baseline random
python scripts/train_attack.py --help
```
Please clear notebook outputs before committing the demo notebooks, to keep diffs readable.
