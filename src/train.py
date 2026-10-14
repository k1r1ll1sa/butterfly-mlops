import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import json
from tqdm import tqdm
from torchvision import models
from prepare_data import prepare_data

torch.set_num_threads(10)


def load_params(path="params.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model(model_name: str, num_classes: int):
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(loader, desc="  Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="  Val", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100.0 * correct / total


def train_model(model_name: str, params: dict):
    train_loader, val_loader, test_loader, classes = prepare_data(params)
    num_classes = len(classes)
    train_cfg = params["train"]
    device = torch.device(train_cfg["device"])

    print(f"\n{'=' * 60}")
    print(f" Обучение модели: {model_name}")
    print(f"{'=' * 60}")

    model = get_model(model_name, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(train_cfg["epochs"]):
        print(f"\nЭпоха {epoch + 1}/{train_cfg['epochs']}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.2f}%")

    models_dir = Path(params["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "num_classes": num_classes,
        "classes": classes
    }, models_dir / f"{model_name}.pth")

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model_name": model_name,
        "final_train_loss": float(train_loss),
        "final_train_acc": float(train_acc),
        "final_val_loss": float(val_loss),
        "final_val_acc": float(val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "history": history
    }

    with open(metrics_dir / f"{model_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


if __name__ == "__main__":
    params = load_params()
    models_to_train = ["resnet18", "efficientnet_b0", "mobilenet_v3_small"]

    all_metrics = {}
    for model_name in models_to_train:
        all_metrics[model_name] = train_model(model_name, params)

    print(f"\n{'=' * 60}")
    print("📊 ИТОГОВАЯ ТАБЛИЦА")
    print(f"{'=' * 60}")
    print(f"{'Модель':<25} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}")
    print("-" * 60)
    for name, m in all_metrics.items():
        print(f"{name:<25} {m['final_train_acc']:>8.2f}%  {m['final_val_acc']:>8.2f}%  {m['test_acc']:>8.2f}%")

    summary_path = Path("metrics") / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nСводные метрики сохранены в: {summary_path}")