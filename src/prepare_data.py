import torch
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml


def load_params(path="params.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_transforms():
    """Базовые трансформации (без аугментации)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform, transform


def prepare_data(params: dict):
    """Создаёт DataLoader'ы для train/val"""
    data_dir = Path(params.get("data_dir", "data/subset"))

    train_cfg = params["train"]
    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(root=data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(root=data_dir / "val", transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"]
    )

    print(f"  DataLoaders созданы для папки: {data_dir}")
    print(f"  Train: {len(train_dataset)} изображений")
    print(f"  Val: {len(val_dataset)} изображений")
    print(f"  Классы: {len(train_dataset.classes)}")

    return train_loader, val_loader, train_dataset.classes


if __name__ == "__main__":
    params = load_params()
    train_loader, val_loader, classes = prepare_data(params)