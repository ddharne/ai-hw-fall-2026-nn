# ai-hw-fall-2026-nn — MNIST digit recognition

Three small neural networks trained on MNIST, with and without image augmentation.

**Rule followed:** models are trained only on the official **train** split (55,000 images for training,
5,000 held out for validation / checkpoint selection). The official **test** split (10,000 images) is used
only once, in `test.py`, for the final numbers.

## Models

| Model | Idea |
|---|---|
| **MLP** | Shallow: 784 → 512 (ReLU, dropout) → 10 |
| **CNN** | 2 conv blocks (Conv-BN-ReLU ×2 + MaxPool), channels 32→64, then FC 128 → 10 |
| **ViT** | 7×7 patches → 16 tokens + [CLS], 4 Transformer encoder layers, dim 64, 4 heads |

## Augmentations (train only)
Random affine (±10° rotation, ±10% shift, 0.9–1.1 scale, shear), elastic distortion (p=0.3), random erasing (p=0.2).
No horizontal flips, since a mirrored digit is not the same digit.

Training: AdamW, OneCycle LR schedule, label smoothing 0.05, batch 128.

## How to run
```bash
pip install -r requirements.txt
./run_all.sh            # trains 6 runs, then tests all of them
# or individually
python train.py --model cnn --epochs 15 --augment
python test.py --ckpt checkpoints/cnn_aug.pt
```

## Results
Full table: [`results/results.md`](results/results.md). Per run: `*_test.json` (accuracy + per-class report),
`*_confusion.png` (test confusion matrix), `*_curves.png` (train/val curves).

| Model | Augmentation | Params | Epochs | Val acc | **Test acc** | Test errors |
|---|---|---:|---:|---:|---:|---:|
| CNN | no | 468,202 | 10 | 99.62% | **99.60%** | 40 |
| CNN | yes | 468,202 | 15 | 99.70% | **99.62%** | 38 |
| MLP | no | 407,050 | 10 | 98.56% | **98.49%** | 151 |
| MLP | yes | 407,050 | 15 | 98.62% | **98.79%** | 121 |
| VIT | no | 139,018 | 10 | 97.72% | **97.75%** | 225 |
| VIT | yes | 139,018 | 15 | 98.40% | **98.62%** | 138 |

## Repo layout
```
data.py      # loaders + augmentations
models.py    # MLP, CNN, ViT
train.py     # training (train split only)
test.py      # evaluation on test split, writes results/
run_all.sh   # full experiment
results/     # test results, plots
```
