import cv2
import albumentations as A
from pathlib import Path
import yaml
from tqdm import tqdm
import shutil


def load_params(path="params.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def augment_data(params: dict):
    subset_dir = Path(params["paths"]["subset"])
    aug_dir = Path(params["paths"]["augmented"])

    if aug_dir.exists():
        shutil.rmtree(aug_dir)

    # Настраиваем пайплайн аугментации с актуальным синтаксисом albumentations
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-20, 20), p=0.5),
        A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.3)
    ])

    train_dir = subset_dir / "train"
    aug_train_dir = aug_dir / "train"

    for class_dir in tqdm(train_dir.iterdir(), desc="Классы"):
        if not class_dir.is_dir():
            continue

        dst_class_dir = aug_train_dir / class_dir.name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy2(img_path, dst_class_dir / img_path.name)

                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                for i in range(2):
                    augmented = transform(image=img)
                    aug_img = augmented['image']
                    aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)

                    aug_name = f"aug_{i}_{img_path.name}"
                    cv2.imwrite(str(dst_class_dir / aug_name), aug_img)

    for split in ["val", "test"]:
        src_split_dir = subset_dir / split
        dst_split_dir = aug_dir / split
        if src_split_dir.exists():
            shutil.copytree(src_split_dir, dst_split_dir)

    print(f"\nАугментация завершена")


if __name__ == "__main__":
    params = load_params()
    augment_data(params)