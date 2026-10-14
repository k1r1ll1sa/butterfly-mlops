import cv2
import albumentations as A
from pathlib import Path
import yaml
from tqdm import tqdm
import shutil


def load_params(path="params.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def copy_dir_contents(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.rglob("*"):
        if item.is_file():
            relative_path = item.relative_to(src_dir)
            dst_file = dst_dir / relative_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst_file)


def augment_data(params: dict):
    subset_dir = Path(params["paths"]["subset"])
    aug_dir = Path(params["paths"]["augmented"])

    if aug_dir.exists():
        shutil.rmtree(aug_dir)

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-20, 20), p=0.5),
        A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.3)
    ])

    train_dir = subset_dir / "train"
    aug_train_dir = aug_dir / "train"

    print(f"Аугментация папки: {train_dir}")

    for class_dir in tqdm(train_dir.iterdir(), desc="Классы"):
        if not class_dir.is_dir():
            continue

        dst_class_dir = aug_train_dir / class_dir.name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy2(img_path, dst_class_dir / img_path.name)

                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                for i in range(2):
                    augmented = transform(image=img)
                    aug_img = augmented['image']
                    aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)

                    aug_name = f"aug_{i}_{img_path.stem}{img_path.suffix}"
                    cv2.imwrite(str(dst_class_dir / aug_name), aug_img)

    for split in ["val", "test"]:
        src_split_dir = subset_dir / split
        dst_split_dir = aug_dir / split

        if src_split_dir.exists():
            src_files = list(src_split_dir.rglob("*"))
            src_files = [f for f in src_files if f.is_file()]
            print(f" Копируем {split}: {len(src_files)} файлов из {src_split_dir}")

            copy_dir_contents(src_split_dir, dst_split_dir)

            dst_files = list(dst_split_dir.rglob("*"))
            dst_files = [f for f in dst_files if f.is_file()]
            print(f"✅ {split} скопирован: {len(dst_files)} файлов в {dst_split_dir}")

            if len(dst_files) == 0:
                print(f"ОШИБКА: {split} не скопирован")
        else:
            pass

    print(f"\n{'=' * 60}")
    print("Статистика аугментированного датасета:")
    print(f"{'=' * 60}")
    for split in ["train", "val", "test"]:
        split_dir = aug_dir / split
        if split_dir.exists():
            files = list(split_dir.rglob("*"))
            files = [f for f in files if f.is_file()]
            classes = [d for d in split_dir.iterdir() if d.is_dir()]
            print(f"  {split}: {len(classes)} классов, {len(files)} файлов")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    params = load_params()
    augment_data(params)