# Investigating the Vulnerabilities of Automatic Speech Recognition (ASR) Models
This is the repo used for my DSO Internship Project (named titularly), also under the National University of Singapore's Student Internship Programme (CP3200: SIP I).

The preprint on arXiv can be found [here](https://arxiv.org/abs/2508.09994)!

## What's in here
We split the project in 3 parts:
* **Attack**: We train the attack using gradient-based methods.
  * *Complete suppression*: maximise P(EOT) at the first token, replicated from [prepend_acoustic_attack](https://github.com/rainavyas/prepend_acoustic_attack).
  * *Partial suppression*: maximise P(EOT) over the first few tokens.
* **Model analysis**: Further analysis of the model and audio
  * Encoder, decoder and cross attention maps.
  * Decoder bypass (logit lens).
  * Causal Mediation Analysis, inspired by [ROME](https://arxiv.org/abs/2202.05262).
  * A differential comparison against a mean-amplitude filler.
  * The snippet's frequency profile.
* **Defences**: 
  * Mu-law compression/decompression
  * Butterworth low-, high- and band-pass filters.

## Repo structure
```
src/asrvuln/
  config.py           paths (data/, outputs/) and constants
  data.py             data loading and preparatio
  models.py           openai-whisper and HF Whisper loaders
  audio.py            audio I/O utils
  prepare.py          prepare methods to attach snippet to audio
  attack.py           gradient-based snippet training
  evaluate.py         empty rate, ASL, BLEU, WER, SNR, STOI
  defences.py         mu-law, Butterworth filters
  analysis/           attention, causal, logit_lens, spectral
  viz.py              plotting utils
  gpu.py              GPU memory utils
notebooks/
  01_attack_and_defence.ipynb   demo: snippet in action
  02_model_analysis.ipynb       demo: conduct secondary analysis
archive/              original notebooks
scripts/
  prepare_data.py     download data (TED-LIUM, release 3)
  train_attack.py     train a snippet
  evaluate_attack.py  evaluate a snippet, optionally against a defence
tests/                CPU smoke tests
data/, outputs/       datasets and generated artefacts (git-ignored)
```

## From scratch
Run these in order from the repo root, inside a fresh venv:
```bash
pip install -r requirements.txt                  # 1. dependencies + this repo (editable)
pytest                                           # 2. sanity check, CPU only, a few seconds
python scripts/download_models.py                # 3. cache Whisper tiny.en + small.en
python scripts/prepare_data.py                   # 4. download TED-LIUM
python scripts/train_attack.py --mode complete   # 5. snippet -> outputs/snippets/complete_snippets.pt (~45 min on a GPU)
python scripts/train_attack.py --mode partial --gap 0.01 --train-n 250 --valid-n 100 --mins-limit 0   # optional
```
Then open the notebooks in `notebooks/`
* Demo 01 runs even without step 5: it trains a quick, weak snippet instead.
* Demo 02 needs step 5.

## Other commands
```bash
python scripts/evaluate_attack.py --snippet outputs/snippets/complete_snippets.pt

python scripts/evaluate_attack.py --snippet outputs/snippets/complete_snippets.pt --defence lowpass --cutoff 7500

python scripts/evaluate_attack.py --baseline random

python scripts/train_attack.py --help
```
