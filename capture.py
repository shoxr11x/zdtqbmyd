# -*- coding: utf-8 -*-
"""
Сбор датасета · снимает кадры с камеры в папку dataset/raw/
────────────────────────────────────────────────────────────
Запуск:
    .venv\\Scripts\\python.exe capture.py ruchka
    .venv\\Scripts\\python.exe capture.py rama --source rtsp://...

Клавиши в окне:
    ПРОБЕЛ — сохранить кадр
    A      — автосъёмка (кадр каждые 0.4 сек) вкл/выкл
    Q      — выход

Как снимать правильно (это важнее количества):
    • предмет держи в руке и ХОДИ по помещению — фон должен меняться;
    • крути предмет: сверху, сбоку, под углом, вверх ногами;
    • разный свет: у окна, при лампе, в тени;
    • разное расстояние: близко и далеко;
    • сними и кадры, где предмет частично перекрыт.
    Цель: 150-300 кадров на класс. С автосъёмкой это 2 минуты.
"""
import argparse
import time
from pathlib import Path

import cv2

RAW = Path(__file__).parent / "dataset" / "raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="имя предмета латиницей, например ruchka")
    ap.add_argument("--source", default="0", help="0 = веб-камера, либо rtsp://...")
    args = ap.parse_args()

    out = RAW / args.name
    out.mkdir(parents=True, exist_ok=True)
    n = len(list(out.glob("*.jpg")))
    print(f"Папка: {out}   (уже есть кадров: {n})")

    src = args.source
    cap = cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Камера не открылась.")
        return

    auto, last = False, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        view = frame.copy()
        cv2.rectangle(view, (0, 0), (view.shape[1], 42), (20, 24, 32), -1)
        cv2.putText(view, f"{args.name}: {n} frames | SPACE=save  A=auto[{'ON' if auto else 'off'}]  Q=quit",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        if auto:
            cv2.circle(view, (view.shape[1] - 24, 21), 8, (60, 220, 120), -1)
        cv2.imshow("capture", view)

        save = False
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            save = True
        if k == ord("a"):
            auto = not auto
        if auto and time.time() - last > 0.4:
            save, last = True, time.time()

        if save:
            n += 1
            cv2.imwrite(str(out / f"{args.name}_{n:04d}.jpg"), frame)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Готово. Всего кадров: {n}  →  {out}")


if __name__ == "__main__":
    main()
