import whisper

import torch
from torch import Tensor

"""
Helpers involving model interaction and loss

General flow is Audio Tensor --> Mel Tensor --> model.forward --> Logits --> Get log probabilities
"""

def audio_to_mel(audio: Tensor) -> Tensor:
    return whisper.pad_or_trim(whisper.log_mel_spectrogram(audio, padding=N_SAMPLES),
                              N_FRAMES)

def audio_to_mel_batch(audio_batch: Tensor) -> Tensor:
    if len(audio_batch.shape) == 1:
        audio_batch = audio_batch.unsqueeze(0)
    return torch.stack([audio_to_mel(audio) for audio in audio_batch])

def mel_to_logits_batch(model: whisper.model.Whisper, mel_batch: Tensor, sot_ids: Tensor) -> Tensor:
    sot_ids = sot_ids.unsqueeze(0).expand(mel_batch.size(0), -1).to(device)
    return model.forward(mel_batch, sot_ids)

def get_loss_batch(logits: Tensor, token_id: Tensor) -> Tensor:
    sf = torch.nn.Softmax(dim=1)
    log_probs = torch.log(sf(logits))
    tgt_probs = log_probs[:,token_id].squeeze()
    return -1 * torch.mean(tgt_probs)