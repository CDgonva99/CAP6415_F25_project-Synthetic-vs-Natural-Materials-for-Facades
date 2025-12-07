"""
Dataset utilities (ImageFolder) and DataLoaders.
All comments in English.
"""
from typing import Tuple, List
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    """Return standard ImageNet-style transforms."""
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

def build_loaders(train_dir: str, val_dir: str, img_size: int,
                  batch_size: int, num_workers: int):
    """
    Create PyTorch DataLoaders for a single source (NAT-only or SYN-only).
    Folder layout must be ImageFolder-compatible:
      train_dir/class_a/*.jpg, train_dir/class_b/*.jpg, ...
      val_dir/class_a/*.jpg,   val_dir/class_b/*.jpg,   ...
    """
    t_train = build_transforms(img_size, train=True)
    t_eval  = build_transforms(img_size, train=False)

    ds_train = datasets.ImageFolder(train_dir, transform=t_train)
    ds_val   = datasets.ImageFolder(val_dir,   transform=t_eval)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return dl_train, dl_val, ds_train.classes

def build_test_loader(test_dir: str, img_size: int,
                      batch_size: int, num_workers: int):
    """Create DataLoader for test split (ImageFolder layout)."""
    t_eval = build_transforms(img_size, train=False)
    ds = datasets.ImageFolder(test_dir, transform=t_eval)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    return dl, ds.classes
