"""Train a model on the MNIST TRAIN split.

Usage:
    python train.py --model cnn --epochs 10
    python train.py --model cnn --epochs 15 --augment
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import get_train_val_loaders
from models import build_model, count_params


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mlp", "cnn", "vit"], required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None, help="default depends on model")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    lr = args.lr or {"mlp": 1e-3, "cnn": 2e-3, "vit": 1e-3}[args.model]
    run_name = f"{args.model}_{'aug' if args.augment else 'noaug'}"

    train_loader, val_loader = get_train_val_loaders(args.batch_size, args.augment)
    model = build_model(args.model).to(device)
    print(f"[{run_name}] device={device} params={count_params(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=args.epochs, steps_per_epoch=len(train_loader))

    history, best_val = [], 0.0
    ckpt_path = Path("checkpoints") / f"{run_name}.pt"
    ckpt_path.parent.mkdir(exist_ok=True)
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        tr_loss, tr_acc = loss_sum / total, correct / total
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=val_loss, val_acc=val_acc))
        print(f"epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val:  # model selection uses validation only, never test
            best_val = val_acc
            torch.save({"model": args.model, "state_dict": model.state_dict(),
                        "args": vars(args), "val_acc": val_acc, "epoch": epoch}, ckpt_path)

    train_time = time.time() - start
    Path("results").mkdir(exist_ok=True)
    with open(Path("results") / f"{run_name}_train_history.json", "w") as f:
        json.dump(dict(run=run_name, params=count_params(model), best_val_acc=best_val,
                       train_time_sec=train_time, args=vars(args), history=history), f, indent=2)
    print(f"[{run_name}] done in {train_time:.0f}s, best val acc {best_val:.4f}, saved {ckpt_path}")


if __name__ == "__main__":
    main()
