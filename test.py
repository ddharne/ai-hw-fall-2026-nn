"""Evaluate trained checkpoints on the MNIST TEST split and write results.

Usage:
    python test.py                     # all checkpoints in ./checkpoints
    python test.py --ckpt checkpoints/cnn_aug.pt
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from data import get_test_loader
from models import build_model, count_params
from train import get_device


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        labels.append(y)
    return torch.cat(preds), torch.cat(labels)


def plot_history(run, out_dir):
    hist_file = out_dir / f"{run}_train_history.json"
    if not hist_file.exists():
        return
    h = json.loads(hist_file.read_text())["history"]
    ep = [e["epoch"] for e in h]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, [e["train_loss"] for e in h], label="train"); ax[0].plot(ep, [e["val_loss"] for e in h], label="val")
    ax[0].set_title(f"{run} loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, [e["train_acc"] for e in h], label="train"); ax[1].plot(ep, [e["val_acc"] for e in h], label="val")
    ax[1].set_title(f"{run} accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out_dir / f"{run}_curves.png", dpi=120); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="*", default=None)
    args = p.parse_args()

    device = get_device()
    ckpts = [Path(c) for c in args.ckpt] if args.ckpt else sorted(Path("checkpoints").glob("*.pt"))
    if not ckpts:
        raise SystemExit("No checkpoints found. Run train.py first.")
    test_loader = get_test_loader()
    out_dir = Path("results"); out_dir.mkdir(exist_ok=True)
    rows = []

    for ck in ckpts:
        state = torch.load(ck, map_location=device)
        model = build_model(state["model"]).to(device)
        model.load_state_dict(state["state_dict"])
        preds, labels = predict(model, test_loader, device)
        acc = (preds == labels).float().mean().item()
        run = ck.stem
        print(f"{run:12s} test accuracy: {acc * 100:.2f}%  ({(preds != labels).sum().item()} errors / {len(labels)})")

        cm = confusion_matrix(labels, preds)
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False)
        ax.set_title(f"{run} — test acc {acc * 100:.2f}%")
        fig.tight_layout(); fig.savefig(out_dir / f"{run}_confusion.png", dpi=120); plt.close(fig)
        plot_history(run, out_dir)

        result = dict(run=run, model=state["model"], augment=state["args"]["augment"],
                      epochs=state["args"]["epochs"], params=count_params(model),
                      best_val_acc=state["val_acc"], test_acc=acc,
                      test_errors=int((preds != labels).sum()),
                      report=classification_report(labels, preds, digits=4, output_dict=True))
        (out_dir / f"{run}_test.json").write_text(json.dumps(result, indent=2))
        rows.append(result)

    rows.sort(key=lambda r: (r["model"], r["augment"]))
    lines = ["| Model | Augmentation | Params | Epochs | Val acc | **Test acc** | Test errors |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['model'].upper()} | {'yes' if r['augment'] else 'no'} | {r['params']:,} | "
                     f"{r['epochs']} | {r['best_val_acc'] * 100:.2f}% | **{r['test_acc'] * 100:.2f}%** | {r['test_errors']} |")
    table = "\n".join(lines)
    (out_dir / "results.md").write_text("# MNIST test results\n\n" + table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
