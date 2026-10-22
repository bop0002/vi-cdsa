"""CLI: python -m src.run_baseline {train,evaluate,summarize,all}."""
import argparse
import json
from pathlib import Path

from src.baseline import train
from src.baseline_evaluation import evaluate, summarize
from src.config import PROCESSED, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "evaluate", "summarize", "all"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.json")
    parser.add_argument("--data-dir", type=Path, default=PROCESSED)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/baseline")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/baseline")
    args = parser.parse_args()
    if args.stage == "all":
        # Check both destinations before starting expensive training.
        for path in (args.model_dir, args.output_dir):
            if path.exists() and any(path.iterdir()):
                parser.error(f"Use an empty destination: {path}")
    if args.stage in ("train", "all"):
        config = json.loads(args.config.read_text(encoding="utf-8"))
        train(args.data_dir, args.model_dir, config)
    if args.stage in ("evaluate", "all"):
        evaluate(args.data_dir, args.model_dir, args.output_dir)
    if args.stage == "summarize":
        summarize(args.output_dir)
    print(f"Completed: {args.stage}", flush=True)


if __name__ == "__main__":
    main()
