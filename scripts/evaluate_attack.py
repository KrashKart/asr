"""
Evaluate a trained snippet (optionally against a defence) on the TED-LIUM test split.

    python scripts/evaluate_attack.py --snippet outputs/snippets/complete_snippets.pt
    python scripts/evaluate_attack.py --snippet outputs/snippets/complete_snippets.pt --defence lowpass --cutoff 7500
    python scripts/evaluate_attack.py --baseline random     # random-noise snippet of the same size
"""
import argparse
from pathlib import Path

import torch

from asrvuln import config, data, defences, evaluate, gpu, models
from asrvuln.prepare import PrepareFront


def build_method(defence: str, size: tuple, cutoff: float):
    if defence == "none":
        return PrepareFront(size)
    if defence == "lowpass":
        return defences.PrepareFrontLowpass(size, cutoff=cutoff)
    if defence == "highpass":
        return defences.PrepareFrontHighpass(size, cutoff=cutoff)
    if defence == "mu":
        return defences.PrepareFrontMu(size)
    if defence == "mu-comp":
        return defences.PrepareFrontMuComp(size)
    raise ValueError(defence)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snippet", type=Path, default=config.SNIPPETS_DIR / "complete_snippets.pt")
    parser.add_argument("--index", type=int, default=-1, help="row of the saved snippets tensor (-1 = best)")
    parser.add_argument("--baseline", choices=["none", "random", "empty"], default="none",
                        help="ignore --snippet and use random noise / no snippet instead")
    parser.add_argument("--snippet-len", type=int, default=config.DEFAULT_SNIPPET_SIZE[1], help="length for --baseline random")
    parser.add_argument("--defence", choices=["none", "lowpass", "highpass", "mu", "mu-comp"], default="none")
    parser.add_argument("--cutoff", type=float, default=7500, help="filter cutoff in Hz")
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--eps", type=float, default=config.DEFAULT_CLAMP_EP)
    parser.add_argument("--test-n", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.baseline == "random":
        snippet = (torch.rand(1, args.snippet_len) * 2 - 1) * args.eps
    elif args.baseline == "empty":
        snippet = torch.empty(1, 0)
    else:
        saved = torch.load(args.snippet, weights_only=True)
        snippet = (saved[args.index] if saved.dim() == 2 else saved).reshape(1, -1)

    model = models.load_whisper(args.model, gpu.get_device())
    test = data.load_tedlium("test", n=args.test_n, seed=args.seed)
    method = build_method(args.defence, (1, max(snippet.size(1), 1)), args.cutoff)
    evaluate.evaluate(model, snippet, method, test, args.eps, position=0)


if __name__ == "__main__":
    main()
