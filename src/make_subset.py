import shutil
import random
import pandas as pd
from pathlib import Path
import yaml
from tqdm import tqdm


def load_params(path="params.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_subset(params: dict):
    raw_dir = Path(params["paths"]["raw"])
    subset_dir = Path(params["paths"]["subset"])
    cfg = params["subset"]

    if subset_dir.exists():
        shutil.rmtree(subset_dir)

    csv_path = raw_dir / "Training_set.csv"
    if not csv_path.exists():
        csv_path = next(raw_dir.glob("*[Tt]rain*.csv"), None)

    if not csv_path:
        raise FileNotFoundError(f"Не найден CSV файл с разметкой в {raw_dir}")

    df = pd.read_csv(csv_path)

    col_names = df.columns.tolist()
    filename_col = next((c for c in col_names if any(x in c.lower() for x in ['file', 'name', 'image', 'id'])),
                        col_names[0])
    label_col = next((c for c in col_names if any(x in c.lower() for x in ['label', 'class', 'species', 'category'])),
                     col_names[1])

    classes_dict = {}
    for _, row in df.iterrows():
        img_name = str(row[filename_col])
        label = str(row[label_col]).strip().replace(" ", "_").replace("/", "_")

        img_path = None
        for search_dir in [raw_dir, raw_dir / "train", raw_dir / "Training"]:
            candidate = search_dir / img_name
            if candidate.exists():
                img_path = candidate
                break

        if img_path and img_path.exists():
            if label not in classes_dict:
                classes_dict[label] = []
            classes_dict[label].append(img_path)

    print(f"Найдено уникальных классов: {len(classes_dict)}")

    if len(classes_dict) == 0:
        raise RuntimeError("Не удалось сопоставить картинки из CSV.")

    random.seed(cfg["seed"])
    selected_classes = random.sample(
        list(classes_dict.keys()),
        min(cfg["n_classes"], len(classes_dict))
    )
    print(f"Выбрано классов: {len(selected_classes)}\n")

    total_copied = 0

    for cls_name in tqdm(selected_classes, desc="Обработка классов"):
        images = classes_dict[cls_name]
        random.shuffle(images)

        n_train = cfg["train_per_class"]
        n_val = cfg["val_per_class"]
        n_test = cfg["test_per_class"]

        train_imgs = images[:n_train]
        val_imgs = images[n_train: n_train + n_val]
        test_imgs = images[n_train + n_val: n_train + n_val + n_test]

        for split, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            dst_dir = subset_dir / split / cls_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                shutil.copy2(img, dst_dir / img.name)
                total_copied += 1

    print(f"\nSubset создан: {total_copied}")


if __name__ == "__main__":
    params = load_params()
    make_subset(params)