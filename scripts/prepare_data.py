"""
Download TED-LIUM release 3 (parquet, sanchit-gandhi/tedlium-data) and save the splits used by this project.

    python scripts/prepare_data.py                 # -> data/tedlium/
    python scripts/prepare_data.py --root /path/to/tedlium --n-train 2000
"""
import argparse

from asrvuln import config, data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(config.TEDLIUM_DIR), help="output directory")
    parser.add_argument("--n-train", type=int, default=2_000, help="number of training examples to keep")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace download cache directory")
    args = parser.parse_args()

    data.prepare_tedlium(root=args.root, n_train=args.n_train, cache_dir=args.cache_dir)
    print(f"Saved TED-LIUM splits to {args.root}")


if __name__ == "__main__":
    main()
