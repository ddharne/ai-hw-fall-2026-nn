#!/usr/bin/env bash
# Trains all 3 models with and without augmentation, then tests them all.
set -e
for m in mlp cnn vit; do
  python train.py --model $m --epochs 10
  python train.py --model $m --epochs 15 --augment
done
python test.py
