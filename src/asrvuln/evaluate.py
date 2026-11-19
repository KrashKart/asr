"""Evaluation of the snippet on test set."""
from typing import Dict, Iterable

import torch
from torch import Tensor
import whisper
from torcheval.metrics.functional import bleu_score, word_error_rate
from torchmetrics.functional.audio import signal_noise_ratio, short_time_objective_intelligibility
from tqdm import tqdm

from .config import SAMPLE_RATE
from .prepare import PrepareMethod

IGNORE_TEXT = "ignore_time_segment_in_scoring"

def evaluate(model: whisper.model.Whisper, snippet: Tensor, prepare_method: PrepareMethod,
             test_dataset: Iterable, clamp_ep: float, position=0, verbose: bool = True) -> Dict[str, float]:
    """
    Evaluates the attack on the test dataset on Empty (rate of empty transcriptions), ASL (Average Sequence
    Length), BLEU score, WER (Word Error Rate), SNR (dB) and STOI. For empty transcriptions, BLEU is defined as 0.

    SNR and STOI compare the attacked audio with the audio if the snippet is silent.
    Higher SNR = quieter perturbation. SNR is very large on empty snippets (no perturbation).
    Higher STOI = more intelligible. STOI is within [0, 1]

    test_dataset : TED-LIUM split (Dataset) or a DataLoader over it; items need "audio" and "text"
    clamp_ep     : Clamp limits (for display)
    position     : Position of application of attack snippet (for display)

    Returns a dict of the averaged metrics.
    """
    if verbose:
        print(f"Clamp: {clamp_ep}\nPrepare Method: {prepare_method.name}\nSnippet Size: {prepare_method.snippet_size}\nPosition: {position}")
    empty_counter = 0
    char_counter = 0
    total_examples = 0
    original_chars = 0
    non_empty = 0
    avg_bleu_score = 0
    wer = 0
    snr = 0
    stoi = 0

    fp16 = str(model.device).startswith("cuda")

    snippet = snippet.to(model.device)
    model.eval()
    pbar = tqdm(test_dataset, total=len(test_dataset), desc="Inference")

    for item in pbar:
        example, answer = item["audio"], item["text"]
        if isinstance(answer, (tuple, list)):
            answer = answer[0]
        if answer == IGNORE_TEXT:
            continue
        if example.dim() == 1:
            example = example.unsqueeze(0)
        example = example.to(model.device)
        attacked_example = prepare_method(snippet, example)
        transcription = model.transcribe(attacked_example.squeeze(), language="en",
                                         condition_on_previous_text=False, fp16=fp16)["text"].strip()

        if not transcription:
            empty_counter += 1
        else:
            non_empty += 1
            avg_bleu_score += bleu_score(transcription, [answer], n_gram=1)
        wer += word_error_rate(transcription, answer)

        # audio-quality metrics: attacked vs the same preparation with a silent snippet
        reference = prepare_method(torch.zeros_like(snippet), example).float().cpu()
        attacked_cpu = attacked_example.float().cpu()
        snr += float(signal_noise_ratio(attacked_cpu, reference))
        stoi += float(short_time_objective_intelligibility(attacked_cpu, reference, fs=SAMPLE_RATE))

        char_counter += len(transcription)
        original_chars += len(answer)
        total_examples += 1
        pbar.set_postfix_str(f"Valid Examples: {total_examples} | Non-empty ASL: {'Undefined' if not non_empty else char_counter / non_empty}")

    pbar.close()

    results = {
        "examples": total_examples,
        "empty_rate": empty_counter / total_examples,
        "asl": char_counter / total_examples,
        "original_asl": original_chars / total_examples,
        "bleu": float(avg_bleu_score / total_examples),
        "wer": float(wer / total_examples),
        "snr_db": snr / total_examples,
        "stoi": stoi / total_examples,
    }
    if verbose:
        print("\n")
        print(f"Total valid examples: {total_examples}")
        print(f"Success rate (Empty): {results['empty_rate']}")
        print(f"Success rate (ASL): {results['asl']} (attacked) out of {results['original_asl']} (original)")
        print(f"Average Bleu Score: {results['bleu']}")
        print(f"Average WER: {results['wer']}")
        print(f"Average SNR (dB): {results['snr_db']}")
        print(f"Average STOI: {results['stoi']}")
    return results

def summarise(results: Dict[str, Dict[str, float]]) -> None:
    """Print a table of several ``evaluate`` results keyed by run name."""
    cols = ["empty_rate", "asl", "bleu", "wer", "snr_db", "stoi"]
    width = max(len(k) for k in results) + 2
    print("".ljust(width) + "".join(c.rjust(12) for c in cols))
    for name, r in results.items():
        print(name.ljust(width) + "".join(f"{r[c]:12.4f}" for c in cols))
