# Investigating the Vulnerabilities of Automatic Speech Recognition (ASR) Models
This is the repo used for my DSO Internship Project (named titularly), also under the National University of Singapore's Student Internship Programme (NUS SIP I, module code CP3200). It contains most of my exploratory work and notebooks used during experimentation.

## arXiv
:construction: In progress!! (I hope it works) :building_construction:

# Repo Breakdown
This repo contains five (5) categories of files:
|         Category        |                Description                |                                                                                                                          Files                                                                                                                         |
|:-----------------------:|:-----------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|          Utilities      |            Utility Python files           |                                                                                                             [```code/utils```](code/utils)                                                                                                             |
| Gradient-based attack   |       Used for gradient-based attack      |                          [```whisper_gradient_multi.ipynb```](code/whisper_gradient_multi.ipynb), [```whisper_gradient.ipynb```](code/whisper_gradient.ipynb) and [```whisper_dsmifgsm.ipynb```](code/whisper_dsmifgsm.ipynb).                         |
|      Model Analysis     | Used for dissection and analysis of model | All ```.ipnyb``` files involving attention analysis (containing "```attn```" or "```attention```" in the filename), [```whisper_causal_analysis.ipynb```](code/whisper_causal_analysis.ipynb) and [```differential.ipynb```](code/differential.ipynb). |
|     Defence Testing     |           Used to test defences           |                                                                                               [```whisper_defence.ipynb```](code/whisper_defence.ipynb).                                                                                               |
|     Other notebooks     |                     -                     |                                                                                      The rest of the notebooks used to play around with the model and audio data.                                                                                      |

# Details
## Gradient-based attack
Attacks on the model involving:
* Attack on complete suppression ([```whisper_gradient.ipynb```](code/whisper_gradient.ipynb)), replicated from [here](https://github.com/rainavyas/prepend_acoustic_attack).
* Attack on partial suppression ([```whisper_gradient_multi.ipynb```](code/whisper_gradient_multi.ipynb)).
* Attack on complete suppression with Dual Space, Momentum Iterative Fast Gradient Sign Method, or DS-MI-FGSM ([```whisper_dsmifgsm.ipynb```](code/whisper_dsmifgsm.ipynb)).

## Model Analysis
* Attention mapping (Encoder, Decoder and Cross-attentions)
* Causal Mediation Analysis, inspired by [this paper](https://arxiv.org/abs/2202.05262) about Rank-One Model Editing ([```whisper_causal_analysis.ipynb```](code/whisper_causal_analysis.ipynb)).
* Differential Visualisation on encoder hidden layers ([```differential.ipynb```](code/differential.ipynb)).

## Defence Testing
* Mu-Law Compression and Compression-Decompression
* Butterworth Low-pass filter

Both are in [```whisper_defence.ipynb```](code/whisper_defence.ipynb)
