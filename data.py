"""MNIST loading. Train split for training/validation, test split ONLY for final testing."""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

MEAN, STD = (0.1307,), (0.3081,)


def get_transforms(augment: bool):
    base = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if not augment:
        return transforms.Compose(base)
    # No horizontal flips: a flipped digit is a different (or invalid) digit.
    return transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.RandomApply([transforms.ElasticTransform(alpha=30.0, sigma=5.0)], p=0.3),
        *base,
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), value=0),
    ])


def get_train_val_loaders(batch_size=128, augment=False, val_size=5000, root="./data", seed=42):
    """Splits the official TRAIN set into train / validation. The test set is never touched here."""
    full_aug = datasets.MNIST(root, train=True, download=True, transform=get_transforms(augment))
    full_clean = datasets.MNIST(root, train=True, download=True, transform=get_transforms(False))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(full_aug), generator=g)
    train_idx, val_idx = idx[val_size:], idx[:val_size]
    train_ds = torch.utils.data.Subset(full_aug, train_idx)
    val_ds = torch.utils.data.Subset(full_clean, val_idx)  # validation is never augmented
    kw = dict(num_workers=2, pin_memory=torch.cuda.is_available())
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw),
            DataLoader(val_ds, batch_size=512, shuffle=False, **kw))


def get_test_loader(batch_size=512, root="./data"):
    test_ds = datasets.MNIST(root, train=False, download=True, transform=get_transforms(False))
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
