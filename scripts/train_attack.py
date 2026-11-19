"""
Train a universal adversarial snippet.

    python scripts/train_attack.py --mode complete
    python scripts/train_attack.py --mode partial --gap 0.01 --train-n 250 --valid-n 100 --mins-limit 0

Saves a tensor of shape (iterations + 1, snippet_len) to outputs/snippets/<mode>_snippets.pt
(by default), last row is the best snippet found.
"""
import argparse
import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from asrvuln import attack, config, data, gpu, models
from asrvuln.prepare import PrepareAtPosition

FORWARDS = {"complete": attack.forward, "partial": attack.forward_multi}

def main() -> None:
    """
    Train an attack snippet.

    Modes (--mode):
        complete : force a completely empty transcript (complete suppression)
        partial  : cut the transcript short (partial suppression)

    Arguments:
        --model        Whisper model used (default tiny.en)
        --snippet-len  snippet length (default 10240, i.e. 0.64s at 16 kHz)
        --position     index where the snippet is inserted into the sample (0 = prepended)
        --eps          max absolute value of the snippet (L-inf clamp)
        --lr           learning rate for AdamW
        --patience     how many consecutive validation-loss non-improvements to tolerate
        --gap          delta validation-loss to treat as convergence (default 0.0)
        --iter-limit   max training iterations (default 30)
        --mins-limit   max training time in minutes (default 45)
        --train-n      number of training examples (default 500)
        --valid-n      number of validation examples (default 150)
        --seed         seed for shuffling before picking the clips (default 1)
        --out          where to save the snippets (default outputs/snippets/<mode>_snippets.pt)
        --tensorboard  log losses and the snippet's spectrogram to outputs/runs/
        --batch-size   training batch size (default 1)
        --valid-batch-size  validation batch size (default 16)
        --amp          bf16 autocast for the Whisper forward/backward
        --track-success  none / valid / all: transcript length per example per iteration (default none),
                       saved to outputs/tensors/<mode>_<split>_success.pt and <mode>_<split>_ids.pt
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=FORWARDS, default="complete", help="complete or partial suppression")
    parser.add_argument("--model", default="tiny.en", help="openai-whisper model name")
    parser.add_argument("--snippet-len", type=int, default=config.DEFAULT_SNIPPET_SIZE[1], help="snippet length (default 10240, i.e. 0.64s at 16 kHz)")
    parser.add_argument("--position", type=int, default=0, help="index where the snippet is inserted into the sample (0 = prepended)")
    parser.add_argument("--eps", type=float, default=config.DEFAULT_CLAMP_EP, help="L-inf clamp on the snippet")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--gap", type=float, default=0.0, help="delta validation-loss to treat as convergence (default 0.0)")
    parser.add_argument("--iter-limit", type=int, default=30, help="max training iterations (default 30)")
    parser.add_argument("--mins-limit", type=int, default=45, help="max training time in minutes (default 45)")
    parser.add_argument("--train-n", type=int, default=500)
    parser.add_argument("--valid-n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None, help="output .pt path (default outputs/snippets/<mode>_snippets.pt)")
    parser.add_argument("--tensorboard", action="store_true", help="log to outputs/runs/")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="training batch size (default 1). >1 is much faster but means fewer, averaged optimizer steps per epoch")
    parser.add_argument("--valid-batch-size", type=int, default=16,
                        help="validation batch size (default 16); doesn't affect results, only memory and speed")
    parser.add_argument("--amp", action="store_true", help="bf16 autocast for the Whisper forward/backward (CUDA, Ampere+)")
    parser.add_argument("--track-success", choices=("none", "valid", "all"), default="none",
                        help="record each example's transcript length every iteration (slow: one full transcription "
                             "per clip). 'valid' = validation set only, 'all' = train too. "
                             "Saved to outputs/tensors/<mode>_<split>_success.pt")
    args = parser.parse_args()

    # initialise model and data splits
    device = gpu.get_device()
    model = models.load_whisper(args.model, device)
    # mel input is always (B, 80, 3000), so let cuDNN pick the fastest conv algorithm once
    torch.backends.cudnn.benchmark = True

    train = data.load_tedlium("train", n=args.train_n, seed=args.seed, with_idx=True)
    valid = data.load_tedlium("validation", n=args.valid_n, seed=args.seed, with_idx=True)
    pin = device == "cuda"
    train_loader = DataLoader(train, batch_size=args.batch_size, collate_fn=data.collate_idx, pin_memory=pin)
    valid_loader = DataLoader(valid, batch_size=args.valid_batch_size, collate_fn=data.collate_idx, pin_memory=pin)

    prepare_method = PrepareAtPosition((1, args.snippet_len), args.position)

    writer = None

    # allow for tensorboard logging
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        stamp = datetime.datetime.now().strftime(f"%Y%m%d-%H%M%S_{args.mode}_{args.snippet_len}_{prepare_method.name}")
        writer = SummaryWriter(log_dir=str(config.RUNS_DIR / stamp), max_queue=5)

    # success tracking
    train_success = {} if args.track_success == "all" else None
    valid_success = {} if args.track_success in ("valid", "all") else None

    best, history, train_success, valid_success = attack.train(
        model, FORWARDS[args.mode], train_loader, valid_loader, prepare_method, writer,
        train_success=train_success, valid_success=valid_success,
        lr=args.lr, iter_limit=args.iter_limit or None, mins_limit=args.mins_limit or None,
        patience=args.patience, clamp_epsilon=args.eps, gap=args.gap, amp=args.amp,
    )

    # save snippet
    out = args.out or config.SNIPPETS_DIR / f"{args.mode}_snippets.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(attack.stack_history(best, history), out)
    print(f"Saved {len(history) + 1} snippets (best last) to {out}")

    # save success tracking, in the format notebooks/1_attack_defence.ipynb loads
    for split, success in (("train", train_success), ("valid", valid_success)):
        if success:
            ids, lengths = attack.success_to_tensor(success)
            config.TENSORS_DIR.mkdir(parents=True, exist_ok=True)
            success_path = config.TENSORS_DIR / f"{args.mode}_{split}_success.pt"
            torch.save(lengths, success_path)
            torch.save(ids, config.TENSORS_DIR / f"{args.mode}_{split}_ids.pt")
            print(f"Saved {split} success ({lengths.shape[0]} examples x {lengths.shape[1]} iterations) to {success_path}")

if __name__ == "__main__":
    main()
