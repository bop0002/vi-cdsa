"""CLI script to train and evaluate PhoBERT for In-Domain & Cross-Domain Sentiment Analysis."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_split
from src.error_analysis import extract_error_cases, format_error_table_markdown
from src.phobert import (
    DEFAULT_MODEL_NAME,
    DynamicPaddingCollator,
    LABEL_NAMES,
    ReviewDataset,
    SentimentPhoBERT,
    compute_class_weights,
    evaluate_model,
    train_phobert,
)


def plot_confusion_matrix(cm: list[list[int]], title: str, save_path: Path):
    """Plot and save confusion matrix as PNG."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
    )
    plt.title(title, fontsize=12, pad=12)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("True Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_experiment(
    model_name: str = DEFAULT_MODEL_NAME,
    num_epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_length: int = 128,
    loss_types: list[str] = ["cross_entropy", "focal"],
    device_str: str = "auto",
    sample_limit: int | None = None,
    output_dir: Path = PROJECT_ROOT / "results",
):
    """Run full experimental matrix for PhoBERT."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"=== Starting PhoBERT Experiment ===")
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Epochs: {num_epochs}, Batch Size: {batch_size}, LR: {lr}, Max Length: {max_length}")
    print(f"Loss Types: {loss_types}")
    if sample_limit:
        print(f"Sample limit per split (for dry-run testing): {sample_limit}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collator = DynamicPaddingCollator(tokenizer=tokenizer, max_length=max_length)

    # 1. Load natural splits
    print("\n--- Loading processed Natural Splits (Variant: full) ---")
    splits: dict[str, pd.DataFrame] = {}
    for d in ["A", "B"]:
        for s in ["train", "val", "test"]:
            key = f"{d}_{s}"
            df = load_split(domain=d, split=s, variant="full")
            if sample_limit and len(df) > sample_limit:
                df = df.sample(n=sample_limit, random_state=42).reset_index(drop=True)
            splits[key] = df
            print(f"Loaded {key}: {len(df)} samples")

    all_results = []
    error_frames = []

    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    models_dir = PROJECT_ROOT / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    for loss_type in loss_types:
        print(f"\n==========================================")
        print(f"  TRAINING & EVALUATION WITH LOSS: {loss_type.upper()}")
        print(f"==========================================")

        for train_domain in ["A", "B"]:
            domain_name = "Domain A (Sách - Tiki)" if train_domain == "A" else "Domain B (Nhà hàng - Foody)"
            print(f"\n>>> Training Model on {domain_name} with {loss_type}...")

            train_df = splits[f"{train_domain}_train"]
            val_df = splits[f"{train_domain}_val"]

            train_ds = ReviewDataset(train_df["text_dl"], train_df["label"])
            val_ds = ReviewDataset(val_df["text_dl"], val_df["label"])

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

            # Class weights for focal loss or weighted CE
            class_weights = compute_class_weights(train_df["label"])
            print(f"Class weights (Neg, Neu, Pos): {class_weights.tolist()}")

            model = SentimentPhoBERT(model_name=model_name, num_classes=3)
            save_checkpoint = models_dir / f"phobert_{train_domain}_{loss_type}.pt"

            def cb(ep, tr_loss, v_met):
                print(f"Epoch {ep}/{num_epochs} - Train Loss: {tr_loss:.4f} | Val Loss: {v_met['loss']:.4f} | Val Macro-F1: {v_met['macro_f1']:.4f}")

            train_phobert(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                lr=lr,
                loss_type=loss_type,
                class_weights=class_weights if loss_type == "focal" else None,
                save_path=save_checkpoint,
                device=device,
                progress_callback=cb,
            )

            # Load best checkpoint
            if save_checkpoint.exists():
                model.load_state_dict(torch.load(save_checkpoint, map_location=device))
                print(f"Loaded best checkpoint from {save_checkpoint.name}")

            # Evaluate on both test sets: In-domain and Cross-domain
            for test_domain in ["A", "B"]:
                test_key = f"{test_domain}_test"
                test_df = splits[test_key].copy()
                test_ds = ReviewDataset(test_df["text_dl"], test_df["label"])
                test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

                scenario = f"{train_domain} -> {test_domain}"
                scenario_type = "In-domain" if train_domain == test_domain else "Cross-domain"

                print(f"\nEvaluating scenario: {scenario} ({scenario_type}) [{loss_type}]...")
                eval_res = evaluate_model(model=model, dataloader=test_loader, device=device, return_details=True)

                record = {
                    "model": "PhoBERT",
                    "loss_type": loss_type,
                    "train_domain": train_domain,
                    "test_domain": test_domain,
                    "scenario": scenario,
                    "type": scenario_type,
                    "macro_f1": eval_res["macro_f1"],
                    "accuracy": eval_res["accuracy"],
                    "macro_precision": eval_res["macro_precision"],
                    "macro_recall": eval_res["macro_recall"],
                    "f1_negative": eval_res["f1_neg"],
                    "f1_neutral": eval_res["f1_neu"],
                    "f1_positive": eval_res["f1_pos"],
                }
                all_results.append(record)
                print(f"--> Macro-F1: {eval_res['macro_f1']:.4f} | Accuracy: {eval_res['accuracy']:.4f} | Neutral-F1: {eval_res['f1_neu']:.4f}")

                # Plot and save confusion matrix
                fig_path = figures_dir / f"cm_phobert_{train_domain}_{test_domain}_{loss_type}.png"
                cm_title = f"Confusion Matrix: PhoBERT ({scenario}) - {loss_type.upper()}"
                plot_confusion_matrix(eval_res["confusion_matrix"], cm_title, fig_path)

                # Error Analysis for Cross-domain scenarios
                if scenario_type == "Cross-domain" and loss_type == "focal":
                    test_df["pred"] = eval_res["preds"]
                    err_df = extract_error_cases(
                        df=test_df,
                        source_domain=train_domain,
                        target_domain=test_domain,
                        model_name=f"PhoBERT ({loss_type})",
                        max_cases_per_type=3,
                        total_cases=12,
                    )
                    error_frames.append(err_df)

    # Calculate Performance Drop
    res_df = pd.DataFrame(all_results)
    print("\n=== SUMMARY RESULTS TABLE ===")
    print(res_df[["scenario", "loss_type", "macro_f1", "accuracy", "f1_neutral"]])

    # Save metrics
    metrics_csv = metrics_dir / "phobert_results.csv"
    res_df.to_csv(metrics_csv, index=False)
    metrics_json = metrics_dir / "phobert_results.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved metrics to {metrics_csv} and {metrics_json}")

    # Combine and save Error Analysis
    if error_frames:
        combined_errors = pd.concat(error_frames, ignore_index=True)
        err_csv = metrics_dir / "error_analysis.csv"
        err_md = metrics_dir / "error_analysis.md"
        combined_errors.to_csv(err_csv, index=False)
        with open(err_md, "w", encoding="utf-8") as f:
            f.write("# ERROR ANALYSIS REPORT (CROSS-DOMAIN PHOBERT)\n\n")
            f.write(format_error_table_markdown(combined_errors))
            f.write("\n")
        print(f"Saved Error Analysis to {err_csv} and {err_md}")

    print("\n=== Experiment Completed Successfully! ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PhoBERT Cross-Domain Experiments")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--device", type=str, default="auto", help="Device: cuda, cpu, or auto")
    parser.add_argument("--sample_limit", type=int, default=None, help="Sample limit for dry-run testing")
    parser.add_argument("--loss_types", nargs="+", default=["cross_entropy", "focal"], help="Loss types to run")

    args = parser.parse_args()
    run_experiment(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        loss_types=args.loss_types,
        device_str=args.device,
        sample_limit=args.sample_limit,
    )
