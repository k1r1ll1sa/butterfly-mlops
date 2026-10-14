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
        raise FileNotFoundError("Не найден Training CSV")

    df_train = pd.read_csv(csv_path)

    test_csv_path = raw_dir / "Testing_set.csv"
    if test_csv_path.exists():
        df_test = pd.read_csv(test_csv_path)
        label_cols = [c for c in df_test.columns if 'label' in c.lower() or 'class' in c.lower()]
        if label_cols:
            print(f"Добавляем данные из: {test_csv_path.name}")
            df_test = df_test.rename(columns={label_cols[0]: 'label', 'filename': 'filename'})
            df_train = pd.concat([df_train, df_test], ignore_index=True)

    col_names = df_train.columns.tolist()
    filename_col = next((c for c in col_names if 'file' in c.lower() or 'name' in c.lower() or 'image' in c.lower()),
                        col_names[0])
    label_col = next((c for c in col_names if 'label' in c.lower() or 'class' in c.lower()), col_names[1])

    print(f"Столбцы: Файл='{filename_col}', Класс='{label_col}'")
    print(f"Всего записей в CSV: {len(df_train)}")

    classes_dict = {}
    for _, row in df_train.iterrows():
        img_name = str(row[filename_col])
        label = str(row[label_col]).strip().replace(" ", "_").replace("/", "_").upper()

        img_path = None
        for search_dir in [raw_dir, raw_dir / "train", raw_dir / "test"]:
            candidate = search_dir / img_name
            if candidate.exists():
                img_path = candidate
                break

        if img_path:
            if label not in classes_dict:
                classes_dict[label] = []
            classes_dict[label].append(img_path)

    print(f"Найдено классов: {len(classes_dict)}")

    random.seed(cfg["seed"])
    selected_classes = random.sample(list(classes_dict.keys()), min(cfg["n_classes"], len(classes_dict)))
    print(f"Выбрано классов: {len(selected_classes)}")

    total_copied = 0

    for cls_name in tqdm(selected_classes, desc="Обработка классов"):
        images = classes_dict[cls_name]
        random.shuffle(images)

        total = len(images)
        n_train = int(total * 0.7)
        n_val = int(total * 0.15)
        n_test = total - n_train - n_val

        train_imgs = images[:n_train]
        val_imgs = images[n_train: n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for split, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            if not split_imgs:
                continue

            dst_dir = subset_dir / split / cls_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                shutil.copy2(img, dst_dir / img.name)
                total_copied += 1

    print(f"Subset создан. Скопировано файлов: {total_copied}")


if __name__ == "__main__":
    params = load_params()
    make_subset(params)