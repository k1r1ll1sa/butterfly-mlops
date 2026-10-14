"""Создание подвыборки датасета бабочек на основе CSV-разметки без изменения разрешения."""
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

    # Очищаем выходную папку
    if subset_dir.exists():
        shutil.rmtree(subset_dir)

    # 1. Читаем CSV с тренировочными данными
    csv_path = raw_dir / "Training_set.csv"
    if not csv_path.exists():
        # Пробуем другие возможные названия
        csv_path = next(raw_dir.glob("*[Tt]rain*.csv"), None)

    if not csv_path:
        raise FileNotFoundError(f"Не найден CSV файл с разметкой в {raw_dir}")

    print(f"📖 Читаем разметку из: {csv_path.name}")
    df = pd.read_csv(csv_path)

    # 2. Умный поиск столбцов с именем файла и классом
    col_names = df.columns.tolist()
    filename_col = next((c for c in col_names if any(x in c.lower() for x in ['file', 'name', 'image', 'id'])),
                        col_names[0])
    label_col = next((c for c in col_names if any(x in c.lower() for x in ['label', 'class', 'species', 'category'])),
                     col_names[1])

    print(f"✅ Найдены столбцы: Имя файла = '{filename_col}', Класс = '{label_col}'")

    # 3. Группируем пути к картинкам по классам
    classes_dict = {}
    for _, row in df.iterrows():
        img_name = str(row[filename_col])
        label = str(row[label_col]).strip().replace(" ", "_").replace("/", "_")  # Санитизация имени папки

        # Ищем картинку: она может быть в папке train, test или в корне raw
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

    print(f"📦 Всего найдено уникальных классов: {len(classes_dict)}")

    if len(classes_dict) == 0:
        raise RuntimeError("Не удалось сопоставить картинки из CSV с файлами на диске. Проверьте структуру data/raw/")

    # 4. Выбираем N случайных классов
    random.seed(cfg["seed"])
    selected_classes = random.sample(
        list(classes_dict.keys()),
        min(cfg["n_classes"], len(classes_dict))
    )
    print(f"🎯 Выбрано классов для subset: {len(selected_classes)}\n")

    total_copied = 0

    # 5. Формируем train / val / test для выбранных классов
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
                # Копируем БЕЗ изменения разрешения (shutil.copy2 сохраняет метаданные)
                shutil.copy2(img, dst_dir / img.name)
                total_copied += 1

    print(f"\n🎉 Subset успешно создан! Всего скопировано файлов: {total_copied}")
    print(f"Проверьте структуру в: {subset_dir.resolve()}")


if __name__ == "__main__":
    params = load_params()
    make_subset(params)