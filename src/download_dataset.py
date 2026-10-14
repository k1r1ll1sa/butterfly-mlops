"""Скачивание Butterfly Specie Classification Dataset через kagglehub."""
import kagglehub
from pathlib import Path


def download():
    # Самый популярный датасет бабочек на Kaggle
    # Если у вас другой — замените slug (например, "phucthaiv02/butterfly-species-classification")
    slug = "phucthaiv02/butterfly-species-classification"

    print(f"Скачиваю датасет: {slug}")
    path = kagglehub.dataset_download(slug)

    # Перемещаем в data/raw
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # kagglehub скачивает в временную папку, копируем оттуда
    import shutil
    src = Path(path)
    dst = raw_dir / src.name

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    print(f"✅ Датасет сохранён в: {dst}")
    print(f"Содержимое: {list(dst.iterdir())[:5]}...")


if __name__ == "__main__":
    download()