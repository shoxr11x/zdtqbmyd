# -*- coding: utf-8 -*-
"""
Обучение своей модели YOLO
──────────────────────────
Перед запуском нужен размеченный датасет в папке dataset/ такого вида:

    dataset/
      data.yaml
      images/train/*.jpg   labels/train/*.txt
      images/val/*.jpg     labels/val/*.txt

Такую структуру отдаёт Roboflow (Export → YOLOv11 → download zip) —
просто распакуй содержимое сюда.

Запуск:
    .venv\\Scripts\\python.exe train.py                  # 80 эпох, yolo11n
    .venv\\Scripts\\python.exe train.py --epochs 150 --model yolo11s.pt

После обучения лучшие веса сами копируются в models/<имя>.pt
и появятся в выпадающем списке в интерфейсе.
"""
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

BASE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(BASE / "dataset" / "data.yaml"))
    ap.add_argument("--model", default="yolo11n.pt", help="с какой модели стартуем")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="parts", help="как назвать обученную модель")
    args = ap.parse_args()

    if not Path(args.data).exists():
        print(f"Нет файла {args.data}")
        print("Сначала размести размеченный датасет в папке dataset/ (см. описание вверху файла).")
        return

    gpu = torch.cuda.is_available()
    print("=" * 58)
    print(f"  Обучение: {args.model} → {args.name}")
    print(f"  Устройство: {torch.cuda.get_device_name(0) if gpu else 'CPU (будет медленно)'}")
    print(f"  Эпох: {args.epochs} · размер кадра: {args.imgsz} · батч: {args.batch}")
    print("=" * 58)

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0 if gpu else "cpu",
        patience=25,            # остановиться, если 25 эпох нет улучшений
        name=args.name,
        project=str(BASE / "runs"),
        # аугментация — учим модель не бояться света, поворота и фона
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.5,
        degrees=15, translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0,
    )

    best = BASE / "runs" / args.name / "weights" / "best.pt"
    if best.exists():
        dst = BASE / "models" / f"{args.name}.pt"
        dst.parent.mkdir(exist_ok=True)
        shutil.copy(best, dst)
        print(f"\nГотово. Модель скопирована: {dst}")
        print("Перезапусти app.py — она появится в списке моделей.")


if __name__ == "__main__":
    main()
