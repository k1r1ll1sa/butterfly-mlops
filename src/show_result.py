import json
from pathlib import Path
import yaml


def load_params(path="params.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def show_results():
    metrics_dir = Path("metrics")
    if not metrics_dir.exists():
        print("❌ Папка metrics/ не найдена")
        return

    summary_path = metrics_dir / "metrics_summary.json"
    if not summary_path.exists():
        print(" Файл metrics_summary.json не найден")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    params = load_params()
    data_dir = params.get("data_dir", "data/subset")

    print(f"РЕЗУЛЬТАТЫ ОБУЧЕНИЯ (Датасет: {data_dir})")
    print(f"{'=' * 80}\n")

    print(f"{'Модель':<25} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Test Loss':<12}")
    print("-" * 80)

    for model_name, model_metrics in metrics.items():
        train_acc = model_metrics.get('final_train_acc', 0)
        val_acc = model_metrics.get('final_val_acc', 0)
        test_acc = model_metrics.get('test_acc', 0)
        test_loss = model_metrics.get('test_loss', 0)

        print(f"{model_name:<25} {train_acc:>8.2f}%  {val_acc:>8.2f}%  {test_acc:>8.2f}%  {test_loss:>10.4f}")

    print(f"\n{'=' * 80}")

if __name__ == "__main__":
    show_results()