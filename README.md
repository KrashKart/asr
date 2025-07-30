# Investigating the Vulnerabilities of Automatic Speech Recognition (ASR) Models
This is the repo used for my DSO Internship Project (named titularly), also under the National University of Singapore's Student Internship Programme (NUS SIP I, module code CP3200). It contains most of my exploratory work and notebooks used during experimentation.

## arXiv
:construction: In progress!! (I hope it works) :building_construction:

# Repo Breakdown
This repo contains five (5) categories of files:
|         Category        |                Description                |                                                                                                                          Files                                                                                                                         |
|:-----------------------:|:-----------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|          utils          |            Utility python files           |                                                                                                             [```code/utils```](code/utils)                                                                                                             |
| Gradient-based training |       Used for gradient-based attack      |                          [```whisper_gradient_multi.ipynb```](code/whisper_gradient_multi.ipynb), [```whisper_gradient.ipynb```](code/whisper_gradient.ipynb) and [```whisper_dsmifgsm.ipynb```](code/whisper_dsmifgsm.ipynb).                         |
|      Model Analysis     | Used for dissection and analysis of model | All ```.ipnyb``` files involving attention analysis (containing "```attn```" or "```attention```" in the filename), [```whisper_causal_analysis.ipynb```](code/whisper_causal_analysis.ipynb) and [```differential.ipynb```](code/differential.ipynb). |
|     Defence Testing     |           Used to test defences           |                                                                                               [```whisper_defence.ipynb```](code/whisper_defence.ipynb).                                                                                               |
|     Other notebooks     |                     -                     |                                                                                      The rest of the notebooks used to play around with the model and audio data.                                                                                      |