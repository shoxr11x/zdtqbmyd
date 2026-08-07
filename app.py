# -*- coding: utf-8 -*-
"""
Vision-сервис · детекция + трекинг + счёт на «воротах»
──────────────────────────────────────────────────────
Что делает:
  1) берёт видео с веб-камеры, IP-камеры (RTSP) или с телефона;
  2) YOLO находит каждый объект в кадре и обводит рамкой  → ДЕТЕКЦИЯ;
  3) ByteTrack ведёт каждый объект между кадрами (даёт ID) → ТРЕКИНГ;
  4) виртуальная линия («ворота») считает пересечения      → СЧЁТ;
  5) отдаёт картинку и цифры в браузер.

Как устроено внутри (это важно для плавности):
  поток №1 — берёт кадр (с камеры или с телефона), рисует ПОСЛЕДНЮЮ известную
             разметку и отдаёт картинку;
  поток №2 — гоняет YOLO по самому свежему кадру и обновляет разметку.
  Видео идёт со скоростью источника и не ждёт нейросеть.

Телефон как камера:
  на телефоне открывается /phone, он снимает своей камерой и шлёт кадры
  по WebSocket, а в ответ получает рамки и рисует их поверх своего видео.
  Работает только по https — Safari не даёт камеру по http (см. make_cert.py).

Запуск:
    .venv\\Scripts\\python.exe app.py
Открыть:
    http://127.0.0.1:8004        — с этого компьютера
    https://<ip компьютера>:8443 — с телефона
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shutil
import socket
import sys
import threading
import time
import webbrowser
import zlib
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse, StreamingResponse)
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from ultralytics import YOLO

BASE = Path(__file__).parent
MODELS_DIR = BASE / "models"
DATASET = BASE / "dataset"   # собранный для обучения датасет: images/, labels/, data.yaml
RAW = DATASET / "raw"        # снятые кадры по папкам-классам, разметка лежит рядом
RUNS = BASE / "runs"         # сюда Ultralytics складывает ход обучения
CERTS = BASE / "certs"
STATE_FILE = BASE / "state.json"     # счётчики: переживают перезапуск
EVENTS_FILE = BASE / "events.jsonl"  # журнал пересечений, строка на событие
# Порты 80, 443 и 8004 заняты MES-системой (nginx и turnstile_gateway в Docker) —
# берём заведомо свободные, чтобы ничего ей не сломать.
PORT = 8010          # http — для этого компьютера, без ругани на сертификат
PORT_HTTP = 8090     # http — только перенаправляет на https
PORT_TLS = 8443      # https — сюда ходит телефон, здесь работает камера
HOSTNAME = "vision"  # имя в локальной сети: vision.local
STALE_AFTER = 2.0    # через сколько секунд молчания источника считать данные протухшими

# чтобы модели скачивались рядом с проектом, откуда бы ни запустили
os.chdir(BASE)

# Python отдаёт управление другому потоку раз в 5 мс. Читающему потоку этого мало:
# пока он ждёт своей очереди, камера успевает выкинуть кадр. Уменьшаем до 1 мс.
sys.setswitchinterval(0.001)

if torch.cuda.is_available():
    # cuDNN подбирает самый быстрый алгоритм свёртки под конкретный размер кадра —
    # у нас он не меняется (модель всегда кормят одним и тем же imgsz), так что
    # есть смысл дать ему подобрать раз и закэшировать, а не гадать заново каждый кадр.
    torch.backends.cudnn.benchmark = True


def _silence_abrupt_disconnects():
    """
    Убирает из консоли простыни «ConnectionResetError: [WinError 10054]».

    Когда браузер рвёт соединение резко — обновили страницу, погас экран
    телефона, ушли из Safari — Windows закрывает сокет мгновенно. asyncio
    после этого пытается вежливо попрощаться с уже мёртвым сокетом, получает
    отказ, и печатает трассировку: перехватить её в этом месте некому.
    Соединение и так закрыто, ничего не ломается — это чистый шум.
    У нас он лезет постоянно, потому что поток видео и WebSocket обрываются
    при каждом обновлении страницы.
    """
    if sys.platform != "win32":
        return
    from asyncio.proactor_events import _ProactorBasePipeTransport

    original = _ProactorBasePipeTransport._call_connection_lost

    def quiet(self, exc):
        try:
            original(self, exc)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass   # клиент уже ушёл — прощаться не с кем

    _ProactorBasePipeTransport._call_connection_lost = quiet


_silence_abrupt_disconnects()

# ── Палитра рамок (RGB). Каждому классу — свой стабильный цвет ────────────────
PALETTE = [
    (62, 166, 255), (55, 214, 122), (255, 176, 32), (255, 92, 92),
    (180, 120, 255), (0, 210, 200), (255, 120, 180), (160, 200, 60),
]


def color_for(i: int) -> tuple[int, int, int]:
    return PALETTE[i % len(PALETTE)]


def bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """OpenCV рисует в BGR, а палитра у нас в RGB."""
    return rgb[2], rgb[1], rgb[0]


# ── Картинки с русскими именами ──────────────────────────────────────────────
# cv2.imread и cv2.imwrite на Windows не открывают пути с кириллицей: путь они
# отдают системе в однобайтовой кодировке, и «dataset/raw/зарядка/…» просто не
# находится (проверено на OpenCV 5 — imread возвращает None, imwrite False).
# А классы у нас называются по-русски. Поэтому файл читаем и пишем сами, а
# OpenCV отдаём уже готовые байты в памяти.
def imread_any(path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_any(path, frame: np.ndarray, quality: int = 92) -> bool:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    try:
        buf.tofile(str(path))
    except OSError:
        return False
    return True


# ── Русские названия для стандартных классов COCO ─────────────────────────────
RU = {
    "person": "человек", "bicycle": "велосипед", "car": "машина", "motorcycle": "мотоцикл",
    "bus": "автобус", "truck": "грузовик", "bottle": "бутылка", "cup": "кружка",
    "fork": "вилка", "knife": "нож", "spoon": "ложка", "bowl": "миска",
    "chair": "стул", "couch": "диван", "bed": "кровать", "tv": "телевизор",
    "laptop": "ноутбук", "mouse": "мышь", "remote": "пульт", "keyboard": "клавиатура",
    "cell phone": "телефон", "microwave": "микроволновка", "oven": "духовка",
    "book": "книга", "clock": "часы", "vase": "ваза", "scissors": "ножницы",
    "teddy bear": "мишка", "toothbrush": "щётка", "backpack": "рюкзак",
    "umbrella": "зонт", "handbag": "сумка", "tie": "галстук", "suitcase": "чемодан",
    "banana": "банан", "apple": "яблоко", "orange": "апельсин", "sandwich": "бутерброд",
    "potted plant": "растение", "dining table": "стол", "toilet": "туалет",
    "sink": "раковина", "refrigerator": "холодильник", "hair drier": "фен",
    "dog": "собака", "cat": "кошка", "bird": "птица",
}


def ru(name: str) -> str:
    return RU.get(name, name)


# ── Цвет детали ──────────────────────────────────────────────────────────────
# Тут нейросеть не нужна и была бы лишней: цвет краски заранее известен из
# задания, камере остаётся только подтвердить. Считаем по тону в системе HSV —
# она отделяет «какой это цвет» от «насколько ярко освещено», поэтому лампы
# и тени сбивают её куда меньше, чем обычные RGB.
#
# OpenCV хранит тон в диапазоне 0..179 (половина градусов круга).
HUES = [
    (0, 9, "красный"), (10, 21, "оранжевый"), (22, 33, "жёлтый"),
    (34, 44, "салатовый"), (45, 85, "зелёный"), (86, 100, "бирюзовый"),
    (101, 125, "синий"), (126, 145, "фиолетовый"), (146, 160, "малиновый"),
    (161, 179, "красный"),
]
# Порог насыщенности подобран замером на живых кадрах, а не на глаз.
# При 60 белую деталь в руке в 26 случаях из 100 объявляло красной: кожа
# по тону красно-оранжевая и достаточно насыщенная, чтобы перебить деталь.
# При 120 остаётся 2 ошибки из 100 — краска насыщена сильно, кожа умеренно.
SAT_MIN = 120     # ниже — считаем оттенком серого, а не цветом
VAL_MIN = 40      # темнее — просто тень, судить о цвете нельзя
INSET = 0.30      # сколько отрезать от краёв рамки: там фон, крюк, рука


def dominant_color(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> dict:
    """
    Какого цвета деталь внутри рамки.

    Берём не всю рамку, а её середину: по краям почти всегда попадает фон,
    крюк или рука, и они утягивают ответ на себя.
    """
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    if bw < 6 or bh < 6:
        return {"name": "?", "share": 0.0}
    cx1 = max(0, int(x1 + bw * INSET)); cx2 = min(w, int(x2 - bw * INSET))
    cy1 = max(0, int(y1 + bh * INSET)); cy2 = min(h, int(y2 - bh * INSET))
    roi = frame[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        return {"name": "?", "share": 0.0}
    # мельчить не жалко: цвет от этого не меняется, а считается мгновенно
    if roi.shape[1] > 64:
        roi = cv2.resize(roi, (64, max(1, int(64 * roi.shape[0] / roi.shape[1]))))

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    colored = (sat >= SAT_MIN) & (val >= VAL_MIN)
    total = hue.size
    share = float(colored.sum()) / total

    if share < 0.25:
        # Насыщенного цвета почти нет — деталь серая, белая или чёрная.
        # Различаем их по яркости, это ровно то, чем они и отличаются.
        v = float(np.median(val))
        name = "чёрный" if v < 60 else ("белый" if v > 185 else "серый")
        return {"name": name, "share": round(1.0 - share, 2)}

    # самый частый тон среди насыщенных пикселей
    counts = np.bincount(hue[colored].ravel(), minlength=180)
    peak = int(counts.argmax())
    name = next((n for lo, hi, n in HUES if lo <= peak <= hi), "?")
    # доля пикселей, попавших в тот же цвет, — насколько ответ уверенный
    lo, hi = next(((lo, hi) for lo, hi, n in HUES if lo <= peak <= hi), (peak, peak))
    same = float(counts[lo:hi + 1].sum()) / max(1, int(colored.sum()))
    return {"name": name, "share": round(same * share, 2)}


# ── Шрифт для подписей (OpenCV не умеет кириллицу, рисуем через PIL) ──────────
def load_font(size: int):
    for f in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
              # в контейнере Windows-шрифтов нет, а встроенный запасной
              # не умеет кириллицу — подписи превратились бы в квадраты
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


FONT = load_font(15)

# Подписи кириллицей рисует PIL — OpenCV кириллицу не умеет. Но гонять ради
# этого весь кадр BGR→RGB→PIL→numpy→BGR, как было раньше, слишком дорого:
# четыре полных прохода по кадру на каждом кадре, тридцать раз в секунду.
# Работа эта идёт в потоке-читателе и почти вся держит GIL, поэтому поток
# распознавания оставался без процессора: замерено 440 мс на кадр там, где
# сама сеть считает 7 мс.
#
# Рисуем текст один раз на маленькой чёрно-белой полоске и запоминаем её.
# Полоска не зависит от цвета рамки, поэтому одна и та же подпись годится
# любому классу, а в кадр она попадает обычным присваиванием по маске.
_LABELS: dict[str, np.ndarray] = {}


def label_mask(text: str) -> np.ndarray:
    """Силуэт подписи: 255 там, где буква. Считается один раз на текст."""
    m = _LABELS.get(text)
    if m is not None:
        return m
    w = max(1, int(FONT.getlength(text)) + 2)
    img = Image.new("L", (w, 19), 0)
    ImageDraw.Draw(img).text((0, 0), text, font=FONT, fill=255)
    m = np.array(img)
    # Словарь не должен расти без предела: подпись содержит проценты
    # уверенности, а они меняются. Тысячи полосок по ~4 КБ — это немного,
    # но пусть будет край.
    if len(_LABELS) > 2000:
        _LABELS.clear()
    _LABELS[text] = m
    return m


class Vision:
    """
    Всё состояние одной камеры: источник, модель, ворота, счётчики.

    Экземпляров может быть несколько — по одному на камеру. Общего между ними
    почти ничего: у каждой свои потоки, свои ворота и свой счёт. Модель тоже
    своя, и это не расточительство, а необходимость: track(persist=True) держит
    память трекера внутри самого объекта модели, и две камеры на одной модели
    путали бы номера деталей друг друга.
    """

    def __init__(self, name: str = "A"):
        self.name = name
        # Камера A пишет в прежний файл — счёт, накопленный до появления второй
        # камеры, не должен пропасть. Остальные заводят себе свой.
        self.state_file = STATE_FILE if name == "A" else BASE / f"state_{name.lower()}.json"
        # замок для готовых кадров и статистики
        self.lock = threading.Lock()
        self.frame_id = 0
        # отдельный замок для сырых кадров: чтение не должно ждать статистику
        self.raw_cond = threading.Condition(threading.Lock())
        self.raw: np.ndarray | None = None
        self.raw_id = 0
        # входящие кадры с телефона
        self.in_cond = threading.Condition(threading.Lock())
        self.in_jpeg: bytes | None = None
        self.in_id = 0
        self.phone_connected = False
        # последняя разметка от нейросети — её рисует поток источника
        self.boxes_lock = threading.Lock()
        self.boxes: list[tuple] = []

        self.thread: threading.Thread | None = None
        self.reader: threading.Thread | None = None
        self.running = False
        self.session = 0      # номер запуска: по нему потоки понимают, что они устарели

        self.model: YOLO | None = None
        # yolo11m, а не лёгкая nano: находит заметно надёжнее, а это главное,
        # когда деталей в кадре много и они мелкие. Цена на RTX 4060 в реальной
        # работе — 32 мс на кадр против 16 у nano, то есть 25 распознаваний
        # в секунду вместо 33. Для конвейера и того, и другого с большим запасом.
        self.model_name = "yolo11m.pt"
        self.model_imgsz = 640     # размер кадра, на котором обучена модель
        self.names: dict[int, str] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.source = "0"            # "0" = веб-камера | "phone" | rtsp://...
        self.mirror = True           # зеркалить картинку (для веб-камеры так привычнее)
        self.stream_width = 960      # ширина картинки в браузере
        self.conf = 0.45             # порог уверенности
        self.line_pos = 0.5          # положение «ворот», 0..1
        self.line_orient = "v"       # 'v' вертикальная | 'h' горизонтальная
        self.only: set[str] = set()  # какие классы считать (пусто = все)

        # съёмка датасета: кадры уходят в dataset/raw/<класс>/
        self.capturing = False
        self.capture_name = "detal"
        self.capture_count = 0
        self.capture_skipped = 0
        self.capture_last = 0.0
        self.capture_src = ""      # источник, с которого шла съёмка
        self.capture_prev = None   # уменьшенная копия последнего сохранённого кадра
        self.capture_gap = 0.12    # не чаще ~8 раз в секунду
        # Насколько кадр должен отличаться от предыдущего сохранённого, чтобы
        # попасть в датасет (среднее отличие яркости, шкала 0-255). Соседние
        # кадры видео почти одинаковы: они раздувают датасет, но ничего не
        # добавляют — сеть просто заучивает их наизусть.
        self.capture_diff = 5.0

        self.jpeg: bytes | None = None
        # FPS считаем по числу кадров за окно времени, а не усреднением 1/dt:
        # если кадр приходит из буфера мгновенно, dt≈0 и среднее улетает в небо
        self.hist_video: deque = deque(maxlen=60)
        self.hist_detect: deque = deque(maxlen=60)
        self.error: str | None = None
        # диагностика, миллисекунды
        self.t_wait = self.t_detect = self.t_draw = 0.0

        # счёт
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: {"in": 0, "out": 0})
        self.live: dict[str, int] = {}
        self.prev_side: dict[int, int] = {}     # track_id -> с какой стороны линии был
        self.seen_at: dict[int, float] = {}     # track_id -> когда видели в последний раз
        self.events: deque = deque(maxlen=60)   # журнал пересечений

        # цвет детали
        self.check_color = True                 # считать цвет каждой найденной детали
        self.expect_color = ""                  # какой цвет ждём по заданию; пусто — не сверяем
        self.colors: dict[str, int] = defaultdict(int)   # сколько какого цвета прошло
        self.color_alarms = 0                   # сколько раз цвет не совпал с заданием
        self._saved_at = 0.0                    # когда в последний раз писали состояние

    # ── сохранение: счёт не должен пропадать при перезапуске ─────────────────
    # Раньше всё жило только в памяти: закрыл окно — смена обнулилась. На заводе
    # первое же отключение света стёрло бы весь учёт.
    def _log_event(self, ev: dict):
        """Каждое пересечение — строкой в файл. Это и история, и будущая подача в MES."""
        try:
            with EVENTS_FILE.open("a", encoding="utf-8") as f:
                # камера в записи обязательна: журнал общий на все камеры, и без
                # неё потом не разобрать, кто именно видел эту деталь
                f.write(json.dumps({**ev, "cam": self.name,
                                    "at": datetime.now().isoformat(timespec="seconds")},
                                   ensure_ascii=False) + "\n")
        except OSError:
            pass        # не смогли записать — счёт всё равно не роняем
        # Сохраняем сразу, без задержки: пересечения редки (на конвейере деталь
        # раз в несколько секунд), а из-за экономии терялось последнее событие
        # перед выключением — проверено, счёт восстанавливался неполным.
        self._save_state(force=True)

    def _save_state(self, force: bool = False):
        """Слепок счётчиков на диск. Не чаще раза в две секунды, чтобы не молотить диск."""
        now = time.time()
        if not force and now - self._saved_at < 2.0:
            return
        self._saved_at = now
        try:
            with self.lock:
                data = {
                    "counts": {k: dict(v) for k, v in self.counts.items()},
                    "colors": dict(self.colors),
                    "color_alarms": self.color_alarms,
                    "events": list(self.events)[:60],
                    # какая модель была выбрана: иначе после перезапуска сервис
                    # молча возвращался к обычной модели, которая своих деталей
                    # не знает — и выглядело это как «перестало распознавать»
                    "model": self.model_name,
                    "expect_color": self.expect_color,
                    "saved": datetime.now().strftime("%d.%m %H:%M:%S"),
                }
            self.state_file.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        except OSError:
            pass

    def load_state(self):
        """Поднимаем счётчики после перезапуска."""
        if not self.state_file.exists():
            return
        try:
            d = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        with self.lock:
            for k, v in (d.get("counts") or {}).items():
                self.counts[k] = {"in": int(v.get("in", 0)), "out": int(v.get("out", 0))}
            for k, v in (d.get("colors") or {}).items():
                self.colors[k] = int(v)
            self.color_alarms = int(d.get("color_alarms") or 0)
            for ev in reversed(d.get("events") or []):
                self.events.appendleft(ev)
        # номера объектов после перезапуска начнутся заново, поэтому память
        # о том, кто с какой стороны линии был, не восстанавливаем — иначе
        # первый же кадр насчитал бы ложных пересечений
        return d.get("saved")

    @staticmethod
    def _rate(hist: deque) -> float:
        """Кадров в секунду по истории моментов времени."""
        if len(hist) < 2:
            return 0.0
        # источник умолк — честнее показать ноль, чем застывшее последнее значение
        if time.perf_counter() - hist[-1] > STALE_AFTER:
            return 0.0
        span = hist[-1] - hist[0]
        return (len(hist) - 1) / span if span > 0 else 0.0

    @property
    def is_phone(self) -> bool:
        return self.source.strip().lower() == "phone"

    # ── модель ────────────────────────────────────────────────────────────────
    def load_model(self, name: str):
        """Своя обученная модель ищется в models/, готовая — скачается сама."""
        local = MODELS_DIR / name
        path = str(local) if local.exists() else name
        self.model = YOLO(path)
        self.model_name = name
        self.names = self.model.names
        # На каком размере кадра модель обучалась. Кормить её кадром другого
        # размера — терять качество: замерено, что модели, обученной на 640,
        # кадр 1280 уронил уверенность с 0.93 до 0.86. Телефон возьмёт это
        # число и будет слать ровно столько, сколько нужно.
        self.model_imgsz = 640
        try:
            tr = (getattr(self.model, "ckpt", None) or {}).get("train_args") or {}
            v = int(tr.get("imgsz") or 640)
            self.model_imgsz = max(320, min(1600, v))
        except (TypeError, ValueError, AttributeError):
            pass
        # Забываем только память трекера: номера объектов от прежней модели
        # ничего не значат. Счёт смены при этом обнулять нельзя — раньше здесь
        # стоял полный сброс, и он затирал не только счётчики в памяти, но и
        # сохранённый файл: каждый запуск камеры стирал учёт.
        self._reset_tracks()

    # ── счётчики ──────────────────────────────────────────────────────────────
    def _reset_tracks(self):
        """Забыть, кто с какой стороны линии был. Счёт не трогаем."""
        with self.lock:
            self.prev_side.clear()
            self.seen_at.clear()

    def reset(self):
        """Полный сброс счёта — только по кнопке «Сбросить», не сам по себе."""
        with self.lock:
            self.counts.clear()
            self.prev_side.clear()
            self.seen_at.clear()
            self.events.clear()
            self.colors.clear()
            self.color_alarms = 0
        # сохранённый слепок тоже обнуляем, иначе после перезапуска
        # счётчики вернулись бы обратно
        self._save_state(force=True)

    # ── запуск / остановка ────────────────────────────────────────────────────
    # У каждого запуска свой номер. Потоки проверяют его на каждом круге и
    # тихо уходят, если номер сменился. Без этого при переключении источника
    # (веб-камера → телефон) старый поток мог пережить остановку и продолжить
    # писать кадры вместе с новым: в датасет попадали кадры двух разных
    # размеров вперемешку, а трекер спотыкался на каждом втором.
    def start(self):
        if self.running:
            return
        if self.model is None:
            self.load_model(self.model_name)
        self.error = None
        self.session += 1
        self.running = True
        me = self.session
        self.thread = threading.Thread(target=self._detect_loop, args=(me,), daemon=True)
        self.thread.start()

    def stop(self):
        self.session += 1        # всё, что работало до этого, теперь устарело
        self.running = False
        for c in (self.raw_cond, self.in_cond):
            with c:
                c.notify_all()          # разбудить всех ожидающих
        if self.thread:
            self.thread.join(timeout=6)
        self.thread = self.reader = None
        with self.boxes_lock:
            self.boxes = []
        self._save_state(force=True)      # остановились — счёт на диск

    def _open_capture(self):
        src = self.source.strip()
        if src.isdigit():
            # Каким бэкендом открывать — не мелочь. Внешнюю камеру Media Foundation
            # заводит шестнадцать секунд: замерено на Logitech BRIO — 15.6 с при
            # автоподборе, 16.0 с при явном MSMF и 1.1 с через DirectShow.
            # Встроенную все открывают одинаково быстро, так что разницу видно
            # только на второй камере — и выглядит она как «камера не работает»,
            # потому что столько никто не ждёт.
            # Раньше тут стоял отказ от DirectShow. Проверил заново: на этой
            # сборке OpenCV он открывает обе камеры и держит 26-27 к/с в трёх
            # прогонах подряд без единого пропуска. Поэтому пробуем его первым,
            # а прежний путь оставляем запасным — на случай камеры, которой
            # DirectShow не подойдёт.
            idx = int(src)
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(idx)
            # MJPG вместо несжатого потока — камера отдаёт заметно ровнее
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            # 960x540: YOLO всё равно ужимает кадр до 640, зато вдвое меньше пикселей
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        else:
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # не копим задержку
        return cap

    # ── кадр с телефона (зовётся из обработчика WebSocket) ────────────────────
    def push_jpeg(self, data: bytes):
        with self.in_cond:
            self.in_jpeg = data
            self.in_id += 1
            self.in_cond.notify_all()

    # ── съёмка датасета ───────────────────────────────────────────────────────
    def capture_start(self, name: str):
        self.capture_name = "".join(c for c in name.strip() if c.isalnum() or c in "-_") or "detal"
        folder = RAW / self.capture_name
        folder.mkdir(parents=True, exist_ok=True)
        # продолжаем нумерацию, а не затираем снятое раньше
        self.capture_count = len(list(folder.glob("*.jpg")))
        self.capture_skipped = 0
        self.capture_last = 0.0
        self.capture_prev = None
        self.capture_src = self.source     # с какого источника начали снимать
        self.capturing = True

    def capture_stop(self):
        self.capturing = False
        self.capture_prev = None

    def _maybe_save(self, frame: np.ndarray):
        now = time.time()
        if now - self.capture_last < self.capture_gap:
            return

        # сравниваем с последним сохранённым по уменьшенной чёрно-белой копии:
        # так «похожесть» считается за доли миллисекунды
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36))
        if self.capture_prev is not None:
            if float(np.mean(cv2.absdiff(small, self.capture_prev))) < self.capture_diff:
                self.capture_skipped += 1   # почти то же самое — не берём
                return

        self.capture_prev = small
        self.capture_last = now
        self.capture_count += 1
        folder = RAW / self.capture_name
        # Буква камеры в имени — обе камеры пишут в одну и ту же папку класса
        # (это тот же класс детали, просто с двух ракурсов), а без этой буквы
        # обе, стартовав съёмку в одну секунду, посчитали бы один и тот же
        # порядковый номер и затёрли бы кадр друг друга.
        imwrite_any(folder / f"{self.capture_name}_{self.name}_{self.capture_count:04d}.jpg", frame)

    # ── общий хвост для любого источника: раздать кадр и отрисовать ───────────
    def _publish(self, frame: np.ndarray):
        # зеркалим ДО всего, чтобы счёт совпадал с тем, что видно на экране
        if self.mirror:
            frame = cv2.flip(frame, 1)

        # В датасет пишем чистый кадр, без рамок — размечать будем сами.
        # Если источник сменился прямо во время съёмки (веб-камера → телефон),
        # съёмку прекращаем: у камер разный размер кадра, и в одной папке они
        # перемешались бы. Трекер на такой смеси работать не сможет — он ведёт
        # рамку только по кадрам одного размера.
        if self.capturing:
            if self.source != self.capture_src:
                self.capture_stop()
                self.error = ("Съёмка остановлена: сменился источник, "
                              "кадры разных камер в один класс не пишем.")
            else:
                self._maybe_save(frame)

        # отдаём сырой кадр нейросети (она работает в своём темпе)
        with self.raw_cond:
            self.raw = frame
            self.raw_id += 1
            self.raw_cond.notify_all()

        # рисуем последнюю известную разметку на копии и публикуем
        ts = time.perf_counter()
        with self.boxes_lock:
            boxes = self.boxes
        shown = self._draw(frame.copy(), boxes)

        h, w = shown.shape[:2]
        if w > self.stream_width:
            k = self.stream_width / w
            shown = cv2.resize(shown, (self.stream_width, int(h * k)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 75])
        self.t_draw = self.t_draw * 0.9 + (time.perf_counter() - ts) * 1000 * 0.1

        if ok:
            with self.lock:
                self.jpeg = buf.tobytes()
                self.frame_id += 1
                self.hist_video.append(time.perf_counter())

    def _alive(self, me: int) -> bool:
        """Работаем, пока сервис запущен И наш запуск ещё не сменился новым."""
        return self.running and self.session == me

    # ── ПОТОК №1а: веб-камера или RTSP ────────────────────────────────────────
    def _read_loop(self, cap, me: int):
        while self._alive(me):
            ok, frame = cap.read()
            if not ok:
                if self._alive(me):
                    self.error = "Кадр не получен — источник отвалился."
                    self.running = False
                break
            if not self._alive(me):
                break        # пока читали кадр, источник переключили — не публикуем
            self._publish(frame)

    # ── ПОТОК №1б: кадры прилетают с телефона ────────────────────────────────
    def _phone_loop(self, me: int):
        last = -1
        while self._alive(me):
            with self.in_cond:
                fresh = self.in_cond.wait_for(lambda: self.in_id != last, timeout=2.0)
                data, last = self.in_jpeg, self.in_id
            if not fresh or data is None or not self._alive(me):
                continue
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            self._publish(frame)

    # ── ПОТОК №2: YOLO по самому свежему кадру ────────────────────────────────
    def _detect_loop(self, me: int):
        cap = None
        if self.is_phone:
            self.mirror = False          # телефон снимает задней камерой, зеркалить не надо
            self.reader = threading.Thread(target=self._phone_loop, args=(me,), daemon=True)
        else:
            cap = self._open_capture()
            if not cap.isOpened():
                if self._alive(me):
                    src = self.source.strip()
                    hint = ("Встроенная обычно 0, подключённая следом — 1, потом 2. "
                            "Ещё камеру может держать другая программа: Zoom, Skype, «Камера»."
                            if src.isdigit() else
                            "Для ссылки rtsp:// проверь адрес, логин и пароль.")
                    self.error = f"Не удалось открыть источник «{src}». {hint}"
                    self.running = False
                cap.release()
                return
            self.reader = threading.Thread(target=self._read_loop, args=(cap, me), daemon=True)
        self.reader.start()

        last_raw = -1
        misses = 0
        while self._alive(me):
            ts = time.perf_counter()
            with self.raw_cond:
                fresh = self.raw_cond.wait_for(lambda: self.raw_id != last_raw, timeout=2.0)
                frame, last_raw = self.raw, self.raw_id
            self.t_wait = self.t_wait * 0.9 + (time.perf_counter() - ts) * 1000 * 0.1
            if not fresh or frame is None:
                # Камера может открыться и не отдать ни одного кадра — так бывает,
                # когда её держит другая программа или когда номер указывает не на
                # то устройство. Раньше это проходило совсем молча: служба «работает»,
                # ошибок нет, а экран пустой, и понять причину было неоткуда.
                # У телефона пустое ожидание — норма: он мог ещё не подключиться.
                if not self.is_phone:
                    misses += 1
                    if misses >= 3 and self._alive(me):
                        self.error = (
                            f"Источник «{self.source}» открылся, но кадров не даёт. "
                            "Скорее всего камеру держит другая программа или номер "
                            "указывает не на то устройство — попробуй соседний номер.")
                        self.running = False
                        break
                continue
            misses = 0

            ts = time.perf_counter()
            boxes, live = self._detect(frame)
            self.t_detect = self.t_detect * 0.9 + (time.perf_counter() - ts) * 1000 * 0.1

            with self.boxes_lock:
                self.boxes = boxes
            with self.lock:
                self.live = live
                self.hist_detect.append(time.perf_counter())

        # Прибираем за собой, но НЕ трогаем чужой запуск: если пока мы
        # заканчивали, уже стартовал новый источник, сбрасывать ему running
        # и обнулять кадр нельзя — он только что начал работать.
        mine = self.reader
        if mine:
            mine.join(timeout=3)            # поток источника отпускает камеру первым
        if cap:
            # Закрывать камеру, пока другой поток сидит внутри cap.read(), нельзя:
            # это чужая память под чтением, и процесс может просто упасть. Такое
            # бывает ровно в одном случае — камера открылась и намертво замолчала.
            # Тогда лучше оставить её занятой до перезапуска службы, чем уронить
            # весь сервис вместе с накопленным счётом.
            if mine and mine.is_alive():
                self.error = ((self.error or "") + " Камера не отвечает — "
                              "чтобы освободить её, перезапусти программу.").strip()
            else:
                cap.release()
        if self.session == me:
            self.running = False
            self.raw = None

    # ── распознавание одного кадра + счёт на воротах ──────────────────────────
    def _detect(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        # persist=True — трекер помнит объекты между кадрами и выдаёт стабильные ID
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            tracker="bytetrack.yaml",
            device=0 if self.device == "cuda" else "cpu",
            # На видеокарте считаем в половинной точности — почти вдвое быстрее
            # почти без потери точности. На CPU half не поддержан, там как был fp32.
            half=self.device == "cuda",
            verbose=False,
        )
        r = results[0]

        line_x = int(w * self.line_pos)
        line_y = int(h * self.line_pos)

        live: dict[str, int] = defaultdict(int)
        out: list[tuple] = []

        b = r.boxes
        if b is not None and b.id is not None:
            xyxy = b.xyxy.cpu().numpy().astype(int)
            ids = b.id.int().cpu().tolist()
            clss = b.cls.int().cpu().tolist()
            confs = b.conf.cpu().tolist()

            for (x1, y1, x2, y2), tid, ci, cf in zip(xyxy, ids, clss, confs):
                name = self.names.get(ci, str(ci))
                if self.only and name not in self.only:
                    continue

                live[name] += 1
                col = dominant_color(frame, int(x1), int(y1), int(x2), int(y2)) \
                    if self.check_color else {"name": "", "share": 0.0}
                out.append((int(x1), int(y1), int(x2), int(y2), ru(name), int(tid),
                            float(cf), int(ci), col["name"]))

                # ── пересечение «ворот» ──
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                side = (1 if cx > line_x else -1) if self.line_orient == "v" else (1 if cy > line_y else -1)
                was = self.prev_side.get(tid)
                self.prev_side[tid] = side
                self.seen_at[tid] = time.time()
                if was is not None and was != side:
                    direction = "in" if side > 0 else "out"
                    # Сверка с заданием: цвет краски известен заранее, камера
                    # лишь подтверждает. Расхождение — повод для аларма.
                    ok = (not self.expect_color or not col["name"]
                          or col["name"] == self.expect_color)
                    with self.lock:
                        self.counts[name][direction] += 1
                        if col["name"]:
                            self.colors[col["name"]] += 1
                            if not ok:
                                self.color_alarms += 1
                        ev = {"t": datetime.now().strftime("%H:%M:%S"),
                              "cls": ru(name), "dir": direction, "id": tid,
                              "color": col["name"], "color_ok": ok}
                        self.events.appendleft(ev)
                    self._log_event(ev)

        # ByteTrack выдаёт всё новые номера, и память о прошедших объектах росла бы
        # бесконечно. Выкидываем тех, кого не видели полминуты, — вернуться они
        # уже не могут, номер им дадут новый.
        if len(self.prev_side) > 400:
            now = time.time()
            for k in [k for k, t in self.seen_at.items() if now - t > 30]:
                self.prev_side.pop(k, None)
                self.seen_at.pop(k, None)

        return out, dict(live)

    # ── рисование: рамки, подписи, «ворота» ──────────────────────────────────
    def _draw(self, frame: np.ndarray, boxes: list[tuple]):
        h, w = frame.shape[:2]

        labels = []
        for x1, y1, x2, y2, name, tid, cf, ci, col in boxes:
            rgb = color_for(ci)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr(rgb), 2)
            cv2.circle(frame, (cx, cy), 4, bgr(rgb), -1)
            text = f"{name} #{tid} · {cf*100:.0f}%" + (f" · {col}" if col else "")
            # ширину плашки берём у шрифта, а не на глаз: раньше стояло
            # 9 пикселей на символ, и у длинных подписей хвост вылезал наружу
            tw = label_mask(text).shape[1]
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), bgr(rgb), -1)
            labels.append((x1 + 5, max(0, y1 - 20), text))

        # «ворота»
        if self.line_orient == "v":
            x = int(w * self.line_pos)
            cv2.line(frame, (x, 0), (x, h), (60, 220, 255), 3)
            cv2.line(frame, (x, 0), (x, h), (255, 255, 255), 1)
        else:
            y = int(h * self.line_pos)
            cv2.line(frame, (0, y), (w, y), (60, 220, 255), 3)
            cv2.line(frame, (0, y), (w, y), (255, 255, 255), 1)

        # Подписи кириллицей: готовую полоску кладём прямо в кадр по маске.
        # Весь кадр через PIL больше не гоняем — см. label_mask().
        for x, y, text in labels:
            m = label_mask(text)
            th, tw = m.shape
            # обрезаем по краю кадра, иначе подпись у самой границы уронит срез
            th, tw = min(th, h - y), min(tw, w - x)
            if th <= 0 or tw <= 0:
                continue
            roi = frame[y:y + th, x:x + tw]
            roi[m[:th, :tw] > 96] = (10, 18, 28)

        return frame

    # ── сводка для интерфейса ─────────────────────────────────────────────────
    def stats(self):
        with self.lock:
            counts = {k: dict(v) for k, v in self.counts.items()}
            events = list(self.events)[:20]
            colors = dict(self.colors)
            live = dict(self.live)
            # источник молчит (телефон свернули, камера отвалилась) — не показываем
            # застывшее «в кадре сейчас», это вводит в заблуждение
            fresh = self.hist_detect and (time.perf_counter() - self.hist_detect[-1] <= STALE_AFTER)
            if not fresh:
                live = {}
        return {
            "running": self.running,
            "error": self.error,
            "fps": round(self._rate(self.hist_video), 1),
            "fps_detect": round(self._rate(self.hist_detect), 1),
            "timing": {"ожидание": round(self.t_wait, 1), "распознавание": round(self.t_detect, 1),
                       "отрисовка": round(self.t_draw, 1)},
            "device": "GPU (CUDA)" if self.device == "cuda" else "CPU",
            "model": self.model_name,
            "model_imgsz": self.model_imgsz,
            "source": self.source,
            "phone": self.phone_connected,
            "mirror": self.mirror,
            "conf": self.conf,
            "line_pos": self.line_pos,
            "line_orient": self.line_orient,
            "only": sorted(self.only),
            "counts": {ru(k): v for k, v in counts.items()},
            "live": {ru(k): v for k, v in live.items()},
            "total_in": sum(v["in"] for v in counts.values()),
            "total_out": sum(v["out"] for v in counts.values()),
            "colors": colors,
            "expect_color": self.expect_color,
            "color_alarms": self.color_alarms,
            "check_color": self.check_color,
            "events": events,
            "classes": sorted(set(self.names.values())) if self.names else [],
        }


# Камеры. «A» — та единственная, что была раньше: её имя стоит по умолчанию
# везде, поэтому старые адреса и сохранённый счёт продолжают работать как были.
# «B» добавляется для съёмки конвейера с двух сторон.
CAMS: dict[str, Vision] = {"A": Vision("A"), "B": Vision("B")}
vision = CAMS["A"]
app = FastAPI(title="Vision · счёт деталей")


def cam(name: str | None = None) -> Vision:
    """Камера по имени из адреса. Неизвестное имя — это «A», а не ошибка:
    половина ссылок в проекте написана вообще без имени."""
    return CAMS.get((name or "A").upper(), CAMS["A"])


class Config(BaseModel):
    source: str | None = None
    conf: float | None = None
    line_pos: float | None = None
    line_orient: str | None = None
    only: str | None = None
    model: str | None = None
    mirror: bool | None = None
    expect_color: str | None = None
    check_color: bool | None = None


# по каким словам в представлении браузера понимаем, что зашли с телефона
MOBILE_UA = ("iphone", "ipod", "ipad", "android", "mobile")


def page(name: str) -> FileResponse:
    """
    Отдать страницу и запретить её кэшировать.

    Без этого браузер держит старую версию: правишь страницу, обновляешь —
    и видишь прежнее. Страницы маленькие, отдаются с этой же машины,
    так что кэшировать их незачем.
    """
    return FileResponse(BASE / "static" / name, headers={
        "Cache-Control": "no-store, must-revalidate",
        "Pragma": "no-cache",
    })


@app.get("/")
def index(request: Request):
    """С телефона сразу отдаём страницу камеры, с компьютера — панель."""
    ua = request.headers.get("user-agent", "").lower()
    if any(m in ua for m in MOBILE_UA) and "desktop" not in request.url.query:
        return RedirectResponse("/phone", status_code=307)
    return page("index.html")


@app.get("/phone")
def phone_page():
    return page("phone.html")


@app.get("/label-phone")
def label_phone_page():
    """Разметка пальцем — рамка с угловыми ручками, как обрезка фото в галерее."""
    return page("label_phone.html")


@app.get("/events")
def events(limit: int = 200):
    """История пересечений из файла — она переживает перезапуск."""
    if not EVENTS_FILE.exists():
        return {"events": []}
    try:
        lines = EVENTS_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {"events": []}
    out = []
    for line in lines[-max(1, min(2000, limit)):]:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    out.reverse()
    return {"events": out, "total": len(lines)}


@app.get("/label")
def label_page():
    """Страница разметки: обводим детали на снятых кадрах."""
    if not (BASE / "static" / "label.html").exists():
        return HTMLResponse("<h1>Страницы разметки нет</h1>"
                            "<p>Файл static/label.html не найден.</p>", status_code=404)
    return page("label.html")


@app.websocket("/ws/camera")
async def ws_camera(ws: WebSocket, cam_name: str = Query("A", alias="cam")):
    """
    Телефон шлёт сюда кадры (JPEG), в ответ получает разметку.
    Рамки рисует сам телефон поверх своего видео — так картинка на нём
    остаётся плавной, по сети летят только координаты.

    Какую камеру обслуживать, решает параметр ?cam= в адресе: страница
    /phone?cam=B подключается сюда как /ws/camera?cam=B. Без него — камера A,
    как и было раньше с одним телефоном.
    """
    await ws.accept()
    v = cam(cam_name)
    v.phone_connected = True
    # если сервис ещё не запущен на телефоне как на источнике — переключаем сами.
    # Обязательно в отдельном потоке: stop() ждёт завершения потоков до 6 секунд,
    # а прямо здесь это заморозило бы весь сервер.
    if not v.running or not v.is_phone:
        def switch():
            v.stop()
            v.source = "phone"
            # пока идёт обучение, видеокарту не трогаем: кадры телефон всё равно
            # шлёт — просто пока без рамок
            if not trainer.running:
                v.start()
        await asyncio.to_thread(switch)
    try:
        while True:
            data = await ws.receive_bytes()
            v.push_jpeg(data)
            with v.boxes_lock:
                boxes = list(v.boxes)
            with v.lock:
                counts = {k: dict(c) for k, c in v.counts.items()}
            await ws.send_json({
                "boxes": [{"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
                           "name": b[4], "id": b[5], "conf": round(b[6], 2), "ci": b[7],
                           "color": b[8]}
                          for b in boxes],
                "line_pos": v.line_pos,
                "line_orient": v.line_orient,
                "in": sum(c["in"] for c in counts.values()),
                "out": sum(c["out"] for c in counts.values()),
                "fps_detect": round(v._rate(v.hist_detect), 1),
                "model_imgsz": v.model_imgsz,
                "shot": shot_orientation_cached(),
                "capturing": v.capturing,
                "capture_name": v.capture_name,
                "capture_count": v.capture_count,
                "capture_skipped": v.capture_skipped,
            })
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        v.phone_connected = False


@app.get("/stream")
async def stream(cam_name: str = Query("A", alias="cam")):
    """
    MJPEG-поток: браузер показывает его обычным <img>.
    Генератор именно async: синхронный uvicorn гоняет через пул потоков,
    и на каждый кадр набегает лишний перескок между потоками — отсюда дёрганье.
    """
    v = cam(cam_name)

    async def gen():
        last = -1
        while True:
            if v.frame_id == last:
                await asyncio.sleep(0.004)      # ждём новый кадр, не занимая поток
                continue
            with v.lock:
                buf, last = v.jpeg, v.frame_id
            if buf:
                # Content-Length обязателен: без него браузер ищет границу кадра
                # перебором байтов, и картинка подрагивает
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(buf)).encode() + b"\r\n\r\n" + buf + b"\r\n")
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stats")
def stats(cam_name: str = Query("A", alias="cam")):
    return JSONResponse(cam(cam_name).stats())


@app.get("/cams")
def cams():
    """
    Сводка по всем камерам разом — чтобы панель не опрашивала каждую отдельно.

    Счёт по камерам НЕ складывается. Камеры стоят по бокам одного конвейера и
    видят одни и те же детали: сумма означала бы, что каждая деталь прошла
    дважды. Показываем показания рядом и расхождение между ними — по нему и
    видно, какая сторона снимает лучше и сколько одна теряет против другой.
    """
    out = {}
    for name, v in CAMS.items():
        s = v.stats()
        out[name] = {
            "running": s["running"], "error": s["error"], "source": s["source"],
            "model": s["model"], "fps": s["fps"], "fps_detect": s["fps_detect"],
            "counts": s["counts"], "total_in": s["total_in"], "total_out": s["total_out"],
            "phone": s["phone"],
            "capturing": v.capturing, "capture_name": v.capture_name,
            "capture_count": v.capture_count, "capture_skipped": v.capture_skipped,
        }
    a, b = CAMS["A"].stats(), CAMS["B"].stats()
    # расхождение считаем по каждому классу отдельно: общая сумма прячет случай,
    # когда одна камера недосчиталась одних деталей, а другая — других
    classes = set(a["counts"]) | set(b["counts"])
    diff = {}
    for k in classes:
        ai = (a["counts"].get(k) or {}).get("in", 0)
        bi = (b["counts"].get(k) or {}).get("in", 0)
        if ai or bi:
            diff[k] = {"a": ai, "b": bi, "spread": abs(ai - bi)}
    return {"cams": out, "diff": diff,
            "both_running": all(v.running for v in CAMS.values())}


class CaptureCfg(BaseModel):
    active: bool
    name: str | None = None
    cam: str | None = None


@app.post("/capture")
def capture(c: CaptureCfg):
    """Съёмка кадров в датасет прямо с телефона."""
    v = cam(c.cam)
    if c.active:
        v.capture_start(c.name or v.capture_name)
    else:
        v.capture_stop()
    return {"capturing": v.capturing, "name": v.capture_name,
            "count": v.capture_count, "skipped": v.capture_skipped}


class CaptureBothCfg(BaseModel):
    active: bool
    name: str | None = None


@app.post("/capture/both")
def capture_both(c: CaptureBothCfg):
    """
    Съёмка сразу с обеих камер — два ракурса одной детали за одно нажатие
    с компьютера. Один запрос вместо двух отдельных на /capture: иначе, если
    один из них не долетит, камеры разъедутся — одна снимает, другая нет.
    """
    a, b = CAMS["A"], CAMS["B"]
    if c.active:
        if not (a.phone_connected and b.phone_connected):
            return fail("Обе камеры должны быть подключены — открой /phone на обоих телефонах.")
        name = c.name or a.capture_name
        a.capture_start(name)
        b.capture_start(name)
    else:
        a.capture_stop()
        b.capture_stop()
    return {
        "capturing": a.capturing and b.capturing,
        "name": a.capture_name,
        "count": a.capture_count + b.capture_count,
        "skipped": a.capture_skipped + b.capture_skipped,
        "cams": {
            "A": {"count": a.capture_count, "skipped": a.capture_skipped},
            "B": {"count": b.capture_count, "skipped": b.capture_skipped},
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
#  ДАТАСЕТ: кадры, разметка, сборка для обучения
# ═════════════════════════════════════════════════════════════════════════════
# Как всё лежит на диске:
#   dataset/raw/<класс>/<класс>_0001.jpg  — снятый кадр
#   dataset/raw/<класс>/<класс>_0001.txt  — его разметка, формат YOLO:
#       «cls x y w h», по строке на рамку, x и y — центр рамки,
#       все четыре числа — доли от 0 до 1, а не пиксели.
#   Пустой .txt — это кадр-негатив: детали на нём нет. Такие кадры тоже нужны,
#   по ним сеть учится не выдумывать деталь там, где её нет.
#   Номер класса (cls) — позиция имени папки в отсортированном списке классов.

CLASSES_FILE = DATASET / "classes.json"


def _registry() -> list[str]:
    try:
        data = json.loads(CLASSES_FILE.read_text(encoding="utf-8"))
        return [str(x) for x in data] if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _save_registry(names: list[str]):
    DATASET.mkdir(parents=True, exist_ok=True)
    CLASSES_FILE.write_text(json.dumps(names, ensure_ascii=False, indent=1), encoding="utf-8")


def _remap_labels(old: list[str], new: list[str]) -> int:
    """
    Переписывает номера классов во всей уже сделанной разметке.
    Зовётся, когда состав классов изменился: рамки исчезнувших классов
    выбрасываем, остальным проставляем новые номера.
    """
    where = {i: new.index(n) for i, n in enumerate(old) if n in new}
    touched = 0
    for d in (sorted(RAW.iterdir()) if RAW.exists() else []):
        if not d.is_dir():
            continue
        for txt in sorted(d.glob("*.txt")):
            boxes, kept, dirty = read_boxes(txt), [], False
            for b in boxes:
                to = where.get(b["cls"])
                if to is None:
                    dirty = True            # класса больше нет — рамка ни к чему
                    continue
                if to != b["cls"]:
                    b["cls"] = to
                    dirty = True
                kept.append(b)
            if dirty:
                write_boxes(txt, kept)
                touched += 1
    return touched


def dataset_classes() -> list[str]:
    """
    Классы и их ПОСТОЯННЫЕ номера.

    Номер класса записан числом в каждый файл разметки. Если брать порядок из
    сортировки папок, то добавление класса на букву «b» сдвинуло бы номера всех
    следующих — и вся сделанная разметка начала бы означать другие детали.
    Молча: обучение прошло бы до конца и выучило перепутанные метки.
    Поэтому порядок хранится в dataset/classes.json — новые классы дописываются
    в конец, номера существующих не меняются. Если класс всё же исчез, разметка
    переписывается под новый порядок, а не остаётся врать.
    """
    on_disk = sorted(d.name for d in RAW.iterdir() if d.is_dir()) if RAW.exists() else []
    old = _registry()
    new = [n for n in old if n in on_disk] + [n for n in on_disk if n not in old]
    if new != old:
        if old:
            _remap_labels(old, new)
        _save_registry(new)
    return new


def safe_name(name: str | None) -> str:
    """
    Имя класса из запроса. Пропускаем только само имя папки — без слэшей и «..»:
    иначе таким «именем» можно было бы уйти по диску куда угодно за пределы датасета.
    """
    n = (name or "").strip().strip(".")
    if not n or n != Path(n).name or "/" in n or "\\" in n or ":" in n:
        raise HTTPException(status_code=400, detail="Недопустимое имя класса")
    return n


def class_dir(name: str | None, must_exist: bool = True) -> Path:
    d = RAW / safe_name(name)
    if must_exist and not d.is_dir():
        raise HTTPException(status_code=404, detail="Такого класса нет")
    return d


def frame_files(d: Path) -> list[Path]:
    """Кадры класса по алфавиту. На этот порядок опираются и автообводка, и сборка."""
    return sorted(d.glob("*.jpg"), key=lambda p: p.name)


_shot_cache: tuple[float, int, str] = (0.0, -1, "")


def shot_orientation() -> str:
    """
    Как держали телефон, когда снимали датасет: "portrait" или "landscape".

    Сеть не умеет распознавать повёрнутую деталь: замерено, что кадр, повёрнутый
    на 90°, роняет находки с 30/30 при уверенности 0.93 до 0.36. Обычного
    отражения слева направо при обучении для этого мало — поворотов там нет.
    Поэтому снимать и работать надо в одном положении, а расхождение показывать
    сразу, пока человек не решил, что модель просто плохая.

    Размеры берём из заголовка JPEG, картинку не разжимаем. Хватает выборки:
    ответ нужен один на весь датасет.

    Ходить по диску прямо в обработчике кадров нельзя — он асинхронный, и на
    это время встаёт весь сервер вместе с видео. Поэтому здесь только счёт, а
    зовут эту функцию из отдельного потока: см. shot_orientation_cached().
    """
    global _shot_cache
    now = time.monotonic()

    files: list[Path] = []
    for name in dataset_classes():
        files += frame_files(RAW / name)
    if not files:
        _shot_cache = (now, 0, "")
        return ""
    if _shot_cache[1] == len(files):
        _shot_cache = (now, _shot_cache[1], _shot_cache[2])
        return _shot_cache[2]

    step = max(1, len(files) // 24)
    tall = wide = 0
    for f in files[::step]:
        try:
            with Image.open(f) as im:
                w, h = im.size
        except (OSError, ValueError):
            continue
        if h > w:
            tall += 1
        elif w > h:
            wide += 1
    seen = tall + wide
    # смешанный датасет — не наше дело выбирать за человека, молчим
    out = "" if not seen else ("portrait" if tall > seen * 0.8 else
                               "landscape" if wide > seen * 0.8 else "")
    _shot_cache = (now, len(files), out)
    return out


_shot_busy = False


def shot_orientation_cached() -> str:
    """
    То же самое, но мгновенно: отдаёт последнее известное значение, а считает
    новое в стороне. Значение меняется только когда доснимают кадры, так что
    отставание на несколько секунд тут ничего не стоит — в отличие от паузы
    в видеопотоке, которую видно сразу.
    """
    global _shot_busy
    if not _shot_busy and time.monotonic() - _shot_cache[0] > 5.0:
        _shot_busy = True

        def refresh():
            global _shot_busy
            try:
                shot_orientation()
            except OSError:
                pass
            finally:
                _shot_busy = False

        threading.Thread(target=refresh, daemon=True).start()
    return _shot_cache[2]


def frame_path(name: str | None, file: str | None) -> Path:
    """Путь к кадру с теми же проверками, что и для имени класса."""
    d = class_dir(name)
    f = (file or "").strip()
    if not f or f != Path(f).name or not f.lower().endswith(".jpg"):
        raise HTTPException(status_code=400, detail="Недопустимое имя кадра")
    p = d / f
    if not p.exists():
        raise HTTPException(status_code=404, detail="Такого кадра нет")
    return p


def fail(msg: str, code: int = 400, **extra):
    """
    Отказ, понятный обеим страницам: панель смотрит на поле error в ответе,
    страница разметки — на код ответа и поле detail. Отдаём и то, и другое.
    """
    return JSONResponse({"ok": False, "error": msg, "detail": msg, **extra}, status_code=code)


def clamp01(v) -> float:
    try:
        return min(1.0, max(0.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def read_boxes(txt: Path) -> list[dict]:
    """Разметка кадра. Файла нет — кадр не размечен; пустой файл — негатив."""
    boxes: list[dict] = []
    if not txt.exists():
        return boxes
    for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            boxes.append({"cls": int(float(p[0])), "x": float(p[1]), "y": float(p[2]),
                          "w": float(p[3]), "h": float(p[4])})
        except ValueError:
            continue        # кривая строка — пропускаем, остальное читаем дальше
    return boxes


def write_boxes(txt: Path, boxes) -> int:
    """Сохраняет разметку. Пустой список — пустой файл, это кадр-негатив."""
    lines = []
    for b in boxes:
        d = b if isinstance(b, dict) else b.model_dump()
        x, y = clamp01(d.get("x")), clamp01(d.get("y"))
        w, h = clamp01(d.get("w")), clamp01(d.get("h"))
        if w <= 0 or h <= 0:
            continue        # схлопнутая рамка — мусор, в датасет не пускаем
        lines.append(f"{max(0, int(d.get('cls', 0)))} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


@app.get("/dataset")
def dataset():
    """Что уже наснято и сколько из этого размечено."""
    out, total_frames, total_labeled = [], 0, 0
    for name in dataset_classes():
        frames = frame_files(RAW / name)
        # размеченным считаем кадр, у которого есть .txt — в том числе пустой,
        # негатив размечен ничуть не меньше остальных
        labeled = sum(1 for f in frames if f.with_suffix(".txt").exists())
        out.append({"name": name, "frames": len(frames), "labeled": labeled})
        total_frames += len(frames)
        total_labeled += labeled
    return {"classes": out, "total_frames": total_frames, "total_labeled": total_labeled}


@app.get("/dataset/classes")
def dataset_class_list():
    """Порядок важен: позиция имени в списке — это и есть номер класса в разметке."""
    return {"classes": dataset_classes()}


class ClassName(BaseModel):
    name: str


@app.post("/dataset/delete")
def dataset_delete(c: ClassName):
    """Удаляет класс целиком — и кадры, и разметку."""
    d = class_dir(c.name)
    for cam_v in CAMS.values():
        if cam_v.capturing and cam_v.capture_name == d.name:
            cam_v.capture_stop()    # нельзя снимать в папку, которой сейчас не станет
    shutil.rmtree(d, ignore_errors=True)
    # состав классов изменился — dataset_classes() увидит это и сам перепишет
    # номера во всей оставшейся разметке
    return {"ok": True, "classes": dataset_classes()}


class ClassRename(BaseModel):
    name: str
    new: str


@app.post("/dataset/rename")
def dataset_rename(c: ClassRename):
    """Переименовывает класс вместе с кадрами — имена файлов не должны разъезжаться с папкой."""
    old = class_dir(c.name)
    new = RAW / safe_name(c.new)
    if new == old:
        return {"ok": True}
    if new.exists():
        return fail("Класс с таким именем уже есть")

    capturing_cams = [cam_v for cam_v in CAMS.values()
                       if cam_v.capturing and cam_v.capture_name == old.name]
    for cam_v in capturing_cams:
        cam_v.capture_stop()
    old.rename(new)
    # список берём заранее, целиком: переименовывать файлы прямо во время обхода папки нельзя
    for f in sorted(new.iterdir()):
        if f.is_file() and f.name.startswith(old.name + "_"):
            try:
                f.rename(new / (new.name + f.name[len(old.name):]))
            except OSError:
                pass        # не вышло с одним файлом — остальные всё равно переименуем
    # Меняем имя ПРЯМО В СПИСКЕ, на том же месте. Иначе старое имя выпадет,
    # новое допишется в конец — и номер класса сменится, а вся его разметка
    # начнёт указывать на соседа.
    reg = _registry()
    if old.name in reg:
        reg[reg.index(old.name)] = new.name
        _save_registry(reg)

    for cam_v in capturing_cams:
        cam_v.capture_start(new.name)
    return {"ok": True, "classes": dataset_classes()}


@app.get("/dataset/frames")
def dataset_frames(name: str):
    """Список кадров класса и того, что из них уже размечено."""
    files = frame_files(class_dir(name))
    return {"files": [f.name for f in files],
            "labeled": [f.name for f in files if f.with_suffix(".txt").exists()]}


@app.get("/dataset/image")
def dataset_image(name: str, file: str):
    """Сам кадр — его показывает страница разметки."""
    return FileResponse(frame_path(name, file), media_type="image/jpeg")


@app.get("/dataset/label")
def dataset_label_get(name: str, file: str):
    return {"boxes": read_boxes(frame_path(name, file).with_suffix(".txt"))}


class Box(BaseModel):
    cls: int = 0
    x: float
    y: float
    w: float
    h: float


class LabelIn(BaseModel):
    name: str
    file: str
    boxes: list[Box] = []


@app.post("/dataset/label")
def dataset_label_set(c: LabelIn):
    """Сохранить разметку кадра. Пустой список рамок — кадр-негатив, так тоже надо."""
    n = write_boxes(frame_path(c.name, c.file).with_suffix(".txt"), c.boxes)
    return {"ok": True, "boxes": n}


class AutoLabelIn(BaseModel):
    name: str
    from_file: str
    box: Box
    count: int = 80
    # Черновик поверх черновика — это нормально: увидел, что рамка съехала,
    # поправил её на этом кадре и повёл трекер дальше заново. Если разметку
    # всё-таки надо поберечь, можно прислать overwrite = false.
    overwrite: bool = True


@app.post("/dataset/autolabel")
def dataset_autolabel(c: AutoLabelIn):
    """
    Черновая обводка: ведёт рамку трекером по следующим кадрам класса.

    Это именно ЧЕРНОВИК. Трекер попадает точно примерно в 7 кадрах из 10,
    остальное человек поправляет руками — но вместо сотни рамок он рисует одну.
    Считается прямо здесь, не в фоне: 80 кадров по ~50 мс — это около четырёх
    секунд ожидания, столько подождать можно.
    """
    d = class_dir(c.name)
    files = frame_files(d)
    names = [f.name for f in files]
    if c.from_file not in names:
        raise HTTPException(status_code=404, detail="Такого кадра нет")
    start = names.index(c.from_file)
    count = max(1, min(300, int(c.count)))   # 300 кадров ≈ 15 секунд, дольше держать запрос нельзя

    first = imread_any(files[start])
    if first is None:
        raise HTTPException(status_code=400, detail="Кадр не читается")
    h, w = first.shape[:2]

    # рамка приходит в долях, а трекеру нужны пиксели
    x1 = int(round((c.box.x - c.box.w / 2) * w))
    y1 = int(round((c.box.y - c.box.h / 2) * h))
    bw, bh = int(round(c.box.w * w)), int(round(c.box.h * h))
    x1, y1 = max(0, min(w - 2, x1)), max(0, min(h - 2, y1))
    bw, bh = max(2, min(w - x1, bw)), max(2, min(h - y1, bh))

    # В OpenCV 5 из трекеров остался только MIL: CSRT и KCF из сборки убрали.
    tracker = cv2.TrackerMIL.create()
    tracker.init(first, (x1, y1, bw, bh))

    labeled, failed, skipped = [], [], []
    for p in files[start + 1: start + 1 + count]:
        frame = imread_any(p)
        if frame is None:
            failed.append(p.name)
            break                     # кадр не прочитался — дальше трекер поедет вслепую
        fh, fw = frame.shape[:2]
        if (fh, fw) != (h, w):
            # Кадр другого размера — так бывает, если посреди съёмки повернуть
            # телефон. Трекеру такое подавать нельзя: он либо возвращает мусор,
            # либо роняет весь запрос ошибкой распределения памяти.
            failed.append(p.name)
            continue
        try:
            ok, rect = tracker.update(frame)
        except cv2.error:
            failed.append(p.name)
            break                     # трекер сломался — дальше только хуже
        if not ok:
            failed.append(p.name)
            break                     # объект потерян, дальше был бы один мусор
        rx, ry, rw, rh = (float(v) for v in rect)
        bx1, by1 = max(0.0, rx), max(0.0, ry)
        bx2, by2 = min(float(fw), rx + rw), min(float(fh), ry + rh)
        if bx2 - bx1 < 2 or by2 - by1 < 2:
            failed.append(p.name)
            break                     # рамка схлопнулась или ушла за край кадра

        txt = p.with_suffix(".txt")
        if txt.exists() and not c.overwrite:
            skipped.append(p.name)    # тут уже есть разметка — чужую работу не затираем
            continue
        write_boxes(txt, [{"cls": c.box.cls,
                           "x": (bx1 + bx2) / 2 / fw, "y": (by1 + by2) / 2 / fh,
                           "w": (bx2 - bx1) / fw, "h": (by2 - by1) / fh}])
        labeled.append(p.name)

    return {"labeled": labeled, "failed": failed, "skipped": skipped}


# Кириллицу в именах файлов OpenCV не осилит (см. imread_any выше), а кадры при
# обучении читает именно он. Поэтому в собранном датасете имена — латиницей.
TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "c",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e",
    "ю": "yu", "я": "ya",
}


def slug(name: str) -> str:
    """Имя класса → безопасное имя файла латиницей."""
    out = []
    for ch in name.lower():
        if ch in TRANSLIT:
            out.append(TRANSLIT[ch])
        elif ch.isascii() and (ch.isalnum() or ch in "-_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "class"


def class_slugs(classes: list[str]) -> dict[str, str]:
    """Латинские имена классов, обязательно разные — иначе кадры перезатрут друг друга."""
    out, used = {}, set()
    for i, n in enumerate(classes):
        s = slug(n)
        if s in used:
            s = f"{s}{i}"
        used.add(s)
        out[n] = s
    return out


class ExportCfg(BaseModel):
    val_ratio: float = 0.2


# Обстановка, а не деталь. Готовая модель узнаёт эти вещи куда увереннее, чем
# незнакомую ей деталь, и без такого списка «самой уверенной рамкой» на каждом
# кадре оказывалась бы клавиатура или монитор на фоне — проверено на живых кадрах.
SCENERY = {
    "person", "keyboard", "tv", "laptop", "dining table", "desk", "chair", "couch",
    "bed", "refrigerator", "oven", "microwave", "sink", "book", "potted plant",
    "window", "door", "monitor",
}


class PredictIn(BaseModel):
    name: str
    model: str | None = None
    conf: float = 0.12
    # Рамка крупнее этой доли кадра — это почти наверняка фон: стол, монитор,
    # клавиатура. Деталь в руке столько места не занимает.
    max_area: float = 0.35
    overwrite: bool = False
    limit: int = 150


@app.post("/dataset/predict")
def dataset_predict(c: PredictIn):
    """
    Черновая разметка готовой моделью.

    Модель не знает наших деталей по имени, но прекрасно видит «тут лежит
    какой-то предмет» и обводит его точно. Класс её ответа нам не важен —
    берём только рамку и подписываем своим классом: в папке всё равно
    один вид детали.

    Работает заметно лучше трекера: трекер накапливает ошибку и уезжает,
    а модель смотрит каждый кадр заново и не помнит прошлых промахов.
    """
    d = class_dir(c.name)
    classes = dataset_classes()
    ci = classes.index(d.name)

    files = frame_files(d)
    todo = [f for f in files if c.overwrite or not f.with_suffix(".txt").exists()]
    todo = todo[: max(1, min(300, int(c.limit)))]
    if not todo:
        return {"labeled": [], "empty": [], "left": 0,
                "note": "Все кадры уже размечены. Чтобы переразметить, включи перезапись."}

    # Пока идёт обучение, видеокарта занята — считаем на процессоре и берём
    # модель полегче, иначе запрос будет висеть минутами.
    busy = trainer.running
    name = c.model or ("yolo11n.pt" if busy else vision.model_name)
    local = MODELS_DIR / name
    model = YOLO(str(local) if local.exists() else name)
    device = "cpu" if busy or vision.device != "cuda" else 0

    labeled, empty = [], []
    for p in todo:
        img = imread_any(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        r = model.predict(img, conf=max(0.02, min(0.9, c.conf)),
                          device=device, verbose=False)[0]

        best, best_conf = None, 0.0
        if r.boxes is not None and len(r.boxes):
            for (x1, y1, x2, y2), cl, cf in zip(r.boxes.xyxy.cpu().numpy(),
                                                r.boxes.cls.int().cpu().tolist(),
                                                r.boxes.conf.cpu().tolist()):
                nm = r.names.get(cl, "")
                if nm in SCENERY:
                    continue                       # это обстановка, а не деталь
                bw, bh = x2 - x1, y2 - y1
                area = (bw * bh) / (w * h)
                if area > c.max_area or area < 0.004:
                    continue                       # слишком крупное (фон) или пылинка
                ratio = max(bw / max(bh, 1), bh / max(bw, 1))
                if ratio > 3.5:
                    continue                       # длинная полоса — край стола, клавиатура, панель
                touching = ((x1 <= 2) + (y1 <= 2) + (x2 >= w - 2) + (y2 >= h - 2))
                if touching >= 2:
                    continue                       # прижата к краям кадра — почти наверняка фон
                if cf > best_conf:
                    best, best_conf = (x1, y1, x2, y2), cf

        txt = p.with_suffix(".txt")
        if best is None:
            # Ничего не нашла — кадр оставляем НЕразмеченным и ничего не трогаем.
            # Записать сюда пустой файл значило бы сказать «детали на кадре нет»,
            # а она есть: сеть училась бы её не видеть. И чужую разметку,
            # если она тут была, тоже не стираем.
            empty.append(p.name)
            continue
        x1, y1, x2, y2 = best
        write_boxes(txt, [{"cls": ci,
                           "x": (x1 + x2) / 2 / w, "y": (y1 + y2) / 2 / h,
                           "w": (x2 - x1) / w, "h": (y2 - y1) / h}])
        labeled.append(p.name)

    left = sum(1 for f in files if not f.with_suffix(".txt").exists())
    return {"labeled": labeled, "empty": empty, "left": left,
            "model": name, "device": "процессор" if device == "cpu" else "видеокарта"}


@app.post("/dataset/export")
def dataset_export(c: ExportCfg):
    """
    Собирает из размеченных кадров датасет в том виде, который понимает YOLO.

    Кадр идёт в проверочные, если crc32 от его имени попал в нужную долю. Это
    не случайный выбор, а всегда один и тот же: пересобрал датасет — проверочные
    кадры остались теми же, и метрики двух обучений можно честно сравнивать.
    Берём только размеченные кадры, включая негативы.
    """
    ratio = min(0.9, max(0.0, float(c.val_ratio)))
    classes = dataset_classes()
    slugs = class_slugs(classes)

    items = []
    for name in classes:
        seen = set()
        for i, img in enumerate(frame_files(RAW / name), 1):
            txt = img.with_suffix(".txt")
            if not txt.exists():
                continue          # неразмеченный кадр обучению только мешает
            tail = "".join(ch for ch in img.stem if ch.isdigit())[-6:] or f"{i:04d}"
            stem = f"{slugs[name]}_{tail}"
            if stem in seen:
                stem = f"{slugs[name]}_{i:04d}"
            seen.add(stem)
            items.append({"img": img, "txt": txt, "stem": stem,
                          "key": zlib.crc32(img.name.encode("utf-8")) % 1000})

    if not items:
        return fail("Ни один кадр не размечен — сначала обведи детали на странице разметки.",
                    train=0, val=0, classes=classes)

    # С одним-двумя кадрами обучение не то что плохое — оно просто не запустится:
    # часть кадров уходит на проверку, обучающая половина остаётся пустой, и
    # Ultralytics падает с невнятным «Error loading data». Лучше сказать прямо.
    MIN_LABELED = 4
    if len(items) < MIN_LABELED:
        return fail(
            f"Размечено всего {len(items)} — для обучения нужно хотя бы {MIN_LABELED} кадра. "
            "На странице разметки клавиша C копирует рамку с предыдущего кадра, "
            "T проводит её трекером по всей серии.",
            train=0, val=0, classes=classes)

    border = int(round(ratio * 1000))
    split = {it["stem"]: ("val" if it["key"] < border else "train") for it in items}
    # без проверочных кадров обучение не запустится, без обучающих — тем более.
    # Если доля оказалась слишком мелкой, отдаём в пустую половину один кадр.
    if len(items) > 1 and all(v == "train" for v in split.values()):
        split[min(items, key=lambda it: it["key"])["stem"]] = "val"
    if len(items) > 1 and all(v == "val" for v in split.values()):
        split[max(items, key=lambda it: it["key"])["stem"]] = "train"

    # старую сборку сносим целиком: иначе в ней останутся кадры, которые уже удалили
    for sub in ("images", "labels"):
        shutil.rmtree(DATASET / sub, ignore_errors=True)
        for part in ("train", "val"):
            (DATASET / sub / part).mkdir(parents=True, exist_ok=True)

    n = {"train": 0, "val": 0}
    dropped = 0
    for it in items:
        part = split[it["stem"]]
        shutil.copyfile(it["img"], DATASET / "images" / part / f"{it['stem']}.jpg")
        # Разметку не копируем, а перезаписываем — заодно вычищаются кривые строки.
        # Рамку с несуществующим номером класса выбрасываем: Ultralytics такой
        # кадр молча пропустит целиком, и человек не поймёт, куда делись данные.
        boxes = [b for b in read_boxes(it["txt"]) if 0 <= b["cls"] < len(classes)]
        dropped += len(read_boxes(it["txt"])) - len(boxes)
        write_boxes(DATASET / "labels" / part / f"{it['stem']}.txt", boxes)
        n[part] += 1

    lines = ["# Собран автоматически из dataset/raw — руками не правь, при сборке перезапишется.",
             f"train: '{(DATASET / 'images' / 'train').as_posix()}'",
             f"val: '{(DATASET / 'images' / 'val').as_posix()}'",
             f"nc: {len(classes)}",
             "names:"]
    for i, name in enumerate(classes):
        lines.append(f"  {i}: '{name.replace(chr(39), chr(39) * 2)}'")
    (DATASET / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Запоминаем состав собранного датасета. Обучение идёт именно по нему, а не
    # по папке с кадрами: доснял деталь и не пересобрал — обучение пойдёт по
    # старому составу и никак об этом не скажет. Панель это показывает.
    per_class = defaultdict(lambda: {"train": 0, "val": 0})
    for part in ("train", "val"):
        for txt in (DATASET / "labels" / part).glob("*.txt"):
            for b in read_boxes(txt):
                if 0 <= b["cls"] < len(classes):
                    per_class[classes[b["cls"]]][part] += 1
    (DATASET / "built.json").write_text(json.dumps({
        "at": datetime.now().strftime("%d.%m %H:%M"),
        "ts": time.time(),
        "classes": classes,
        "train": n["train"], "val": n["val"],
        "per_class": {k: dict(v) for k, v in per_class.items()},
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    return {"ok": True, "train": n["train"], "val": n["val"],
            "classes": classes, "dropped": dropped}


@app.get("/dataset/built")
def dataset_built():
    """
    Что лежит в СОБРАННОМ датасете — именно на нём пойдёт обучение.
    Отдельно говорим, менялась ли разметка после сборки: это самая
    незаметная ошибка — доснять деталь и забыть пересобрать.
    """
    f = DATASET / "built.json"
    if not f.exists():
        return {"built": False, "stale": False,
                "note": "Датасет ещё не собран — нажми «Собрать датасет»."}
    try:
        info = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"built": False, "stale": False, "note": "Сводка о сборке испорчена, собери заново."}

    built_ts = float(info.get("ts") or 0)
    newest, changed = 0.0, []
    for name in dataset_classes():
        d = RAW / name
        for p in list(d.glob("*.txt")) + list(d.glob("*.jpg")):
            t = p.stat().st_mtime
            if t > newest:
                newest = t
            if t > built_ts and name not in changed:
                changed.append(name)

    info["built"] = True
    info["stale"] = newest > built_ts + 1
    info["changed"] = changed
    # чего в сборке нет вовсе — например, класс без единой размеченной рамки
    info["missing"] = [n for n in dataset_classes() if n not in (info.get("per_class") or {})]
    return info


# ═════════════════════════════════════════════════════════════════════════════
#  ОБУЧЕНИЕ СВОЕЙ МОДЕЛИ
# ═════════════════════════════════════════════════════════════════════════════
class TrainLog(logging.Handler):
    """Ultralytics всё рассказывает через свой логгер — подслушиваем и копим строки."""

    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def emit(self, record):
        try:
            text = record.getMessage()
        except Exception:
            return
        for line in text.splitlines():
            line = line.strip()
            if line:
                self.sink(line)


class Trainer:
    """
    Обучение идёт в отдельном потоке: запрос возвращается сразу, а как идут дела —
    видно в /train/status. Страницу можно закрыть, обучение от этого не прервётся.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.running = False
        self.stopping = False        # нажали «остановить» — скажем об этом Ultralytics в колбэке
        self.epoch = 0
        self.epochs = 0
        self.metrics: dict = {}
        self.log: deque = deque(maxlen=200)
        self.done = False
        self.result_model: str | None = None
        self.error: str | None = None

    def say(self, line: str):
        with self.lock:
            self.log.append(line)

    def status(self):
        with self.lock:
            # Подстраховка: любое «не-число» среди метрик обрушило бы весь ответ
            # ошибкой 500, и панель ослепла бы целиком.
            metrics = {k: v for k, v in self.metrics.items()
                       if isinstance(v, (int, float)) and math.isfinite(v)}
            return {"running": self.running, "epoch": self.epoch, "epochs": self.epochs,
                    "metrics": metrics, "log": list(self.log),
                    "done": self.done, "result_model": self.result_model,
                    "error": self.error, "stopping": self.stopping}

    def start(self, cfg) -> dict:
        if self.running:
            return {"ok": False, "error": "Обучение уже идёт"}
        data = DATASET / "data.yaml"
        if not data.exists():
            return {"ok": False, "error": "Датасет не собран — сначала нажми «Собрать датасет»."}
        # имя модели станет именем папки и файла, поэтому только латиница
        name = "".join(ch for ch in cfg.name.strip()
                       if ch.isascii() and (ch.isalnum() or ch in "-_")).strip("-_")
        if not any(ch.isalnum() for ch in name):
            name = "parts"
        with self.lock:
            self.running = True
            self.stopping = False
            self.done = False
            self.error = None
            self.result_model = None
            self.epoch = 0
            self.epochs = max(1, min(1000, int(cfg.epochs)))
            self.metrics = {}
            self.log.clear()
        self.thread = threading.Thread(target=self._run, args=(cfg, name, data), daemon=True)
        self.thread.start()
        return {"ok": True, "name": name}

    def stop(self) -> dict:
        with self.lock:
            was = self.running
            if was:
                self.stopping = True
        if was:
            self.say("Просили остановиться — доучим текущую эпоху и выйдем.")
        return {"ok": True}

    # ── колбэки Ultralytics: только они знают, на какой мы эпохе ──────────────
    def _on_epoch_end(self, trainer):
        with self.lock:
            self.epoch = int(getattr(trainer, "epoch", 0)) + 1
            stop = self.stopping
        if stop:
            # Ultralytics проверяет этот флаг в конце эпохи и выходит из цикла сам.
            # Обрывать поток силой нельзя: недописанные веса потом не откроются.
            trainer.stop = True

    def _on_fit_epoch_end(self, trainer):
        m = {}
        for k, v in (getattr(trainer, "metrics", None) or {}).items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            # Когда обучение разваливается, метрика становится «не-число».
            # Такое значение нельзя положить в JSON — сервер отвечал ошибкой 500,
            # и панель переставала показывать вообще что-либо. Пропускаем.
            if not math.isfinite(fv):
                continue
            m[str(k)] = round(fv, 4)
        with self.lock:
            if m:
                self.metrics = m
            ep, total = self.epoch, self.epochs
        short = {k.split("/")[-1].split("(")[0]: v for k, v in m.items()}
        good = " · ".join(f"{k} {v}" for k, v in short.items() if k in ("mAP50", "mAP50-95"))
        self.say(f"эпоха {ep}/{total}" + (f" · {good}" if good else ""))

    def _report_quality(self, name: str):
        """
        Честный разбор итога по журналу обучения.

        Ultralytics сохраняет «лучшие» веса по своей оценке, но если обучение
        развалилось, лучшими он может счесть уже испорченные. Поэтому смотрим
        сами: какой был максимум, на какой эпохе и чем всё кончилось.
        """
        import csv
        f = RUNS / name / "results.csv"
        if not f.exists():
            return
        try:
            rows = list(csv.DictReader(f.open(encoding="utf-8")))
        except (OSError, ValueError):
            return
        if not rows:
            return

        def val(row, part):
            for k, v in row.items():
                if part in k:
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return 0.0
            return 0.0

        maps = [val(r, "mAP50(B)") for r in rows]
        best = max(maps)
        best_ep = maps.index(best) + 1
        last = maps[-1]

        self.say(f"Итог: лучшая mAP50 = {best:.3f} на эпохе {best_ep} из {len(rows)}.")
        if best > 0.05 and last < best * 0.5:
            self.say("⚠ Обучение развалилось под конец: оценка упала более чем вдвое "
                     "от достигнутой. Модель ненадёжна — обучи заново, уменьшив число эпох "
                     f"примерно до {max(10, best_ep + 5)}.")
        elif best < 0.3:
            self.say("⚠ Оценка низкая. Чаще всего дело в разметке: проверь, что рамки "
                     "обтягивают деталь целиком и одинаково на всех кадрах.")
        elif best < 0.7:
            self.say("Модель рабочая, но неуверенная. Помогут ещё 2-3 съёмки "
                     "в других местах и при другом свете.")
        else:
            self.say("Модель хорошая — можно проверять камерой.")

    # ── сам поток обучения ───────────────────────────────────────────────────
    def _run(self, cfg, name: str, data: Path):
        handler = TrainLog(self.say)
        handler.setLevel(logging.INFO)
        log = logging.getLogger("ultralytics")
        log.addHandler(handler)
        # видеокарта одна: пока учимся, распознавать нечем.
        # Запоминаем, работал ли источник, чтобы вернуть его после обучения —
        # иначе телефон навсегда останется с картинкой без рамок и «YOLO 0/с».
        was_running = vision.running
        try:
            vision.stop()
            self.say("Источник остановлен: камера и распознавание выключены, "
                     "видеокарта отдана обучению целиком.")
            self.say(f"Старт: {cfg.model} → {name}, эпох {self.epochs}, "
                     f"кадр {cfg.imgsz}, батч {cfg.batch}")

            base = MODELS_DIR / cfg.model
            model = YOLO(str(base) if base.exists() else cfg.model)
            model.add_callback("on_train_epoch_end", self._on_epoch_end)
            model.add_callback("on_fit_epoch_end", self._on_fit_epoch_end)

            # ── настройки под размер датасета ────────────────────────────────
            # На сотне кадров и на десяти тысячах нужны РАЗНЫЕ настройки, и это
            # не мелочь: на первом обучении здесь стояли настройки для большого
            # датасета, и обучение развалилось на 30-й эпохе — штраф за класс
            # подскочил с 2.9 до 60, веса разрушились и не восстановились.
            # Настройки обучения — библиотечные, кроме одной.
            #
            # Я пробовал подкручивать шаг обучения, разогрев и аугментацию под
            # маленький датасет — и сделал заметно хуже: mAP50 на 14-й эпохе
            # вышла 0.007 против 0.176 на пятой эпохе с настройками по
            # умолчанию. Настройки Ultralytics подобраны авторами на множестве
            # задач, и менять их без замеров нельзя.
            #
            # Единственное отступление — мозаика на маленьком датасете.
            # Она склеивает четыре кадра в один, объекты становятся вдвое мельче;
            # на сотне кадров это отнимает половину примеров. Отключение мозаики
            # для маленьких датасетов — общепринятая практика, а не догадка.
            n_train = len(list((DATASET / "images" / "train").glob("*.jpg")))
            small = n_train < 300

            # Тяжёлая модель на маленьком датасете не учится, а срывается.
            # Проверено на живых данных: yolo11m (20 млн параметров) на 78 кадрах
            # трижды разваливалась на 27-30 эпохе — штраф за класс подскакивал
            # вчетверо и оценка падала в ноль. yolo11n (2.6 млн) на тех же кадрах
            # дошла до mAP50 0.995 без единого срыва.
            if n_train and ((cfg.model == "yolo11m.pt" and n_train < 1000)
                            or (cfg.model == "yolo11s.pt" and n_train < 300)):
                self.say(f"⚠ Кадров всего {n_train}, а модель {cfg.model} тяжёлая. "
                         "Обучение может развалиться на середине — для такого датасета "
                         "надёжнее yolo11n.pt.")

            # Чем крупнее кадр при обучении, тем больше памяти нужно видеокарте.
            # Подбираем батч под размер сами, иначе на 1280 обучение падает
            # с нехваткой памяти, а человек видит невнятную ошибку.
            imgsz = max(320, min(1280, int(cfg.imgsz)))
            batch = int(cfg.batch) if cfg.batch else 0
            if not batch:
                batch = 16 if imgsz <= 640 else (8 if imgsz <= 960 else 4)
            if imgsz > 640:
                self.say(f"Кадр {imgsz} — крупнее обычного: мелкие детали видны лучше, "
                         f"но обучение идёт дольше. Батч подобран {batch}.")

            model.train(
                data=str(data),
                epochs=self.epochs,
                imgsz=imgsz,
                batch=batch,
                device=0 if vision.device == "cuda" else "cpu",
                patience=30,                 # 30 эпох без улучшений — дальше смысла нет
                project=str(RUNS), name=name, exist_ok=True,
                mosaic=0.0 if small else 1.0,
            )
            if small:
                self.say("Датасет небольшой — мозаика выключена, остальное по умолчанию.")
            self._report_quality(name)

            best = RUNS / name / "weights" / "best.pt"
            if best.exists():
                MODELS_DIR.mkdir(exist_ok=True)
                shutil.copyfile(best, MODELS_DIR / f"{name}.pt")
                with self.lock:
                    self.result_model = f"{name}.pt"
                self.say(f"Готово. Веса лежат в models/{name}.pt — модель уже в списке.")
            else:
                self.say("Обучение закончилось, а файла весов нет — смотри лог выше.")
        except Exception as e:
            # поток не должен падать молча: иначе в интерфейсе всё «идёт», а на деле стоит
            with self.lock:
                self.error = f"{type(e).__name__}: {e}"
            self.say(f"Ошибка: {e}")
        finally:
            log.removeHandler(handler)
            with self.lock:
                self.running = False
                self.stopping = False
                self.done = True
            # видеокарта освободилась — поднимаем источник обратно, если он работал
            if was_running:
                try:
                    vision.start()
                    self.say("Источник запущен обратно — распознавание снова работает.")
                except Exception as e:
                    self.say(f"Источник поднять не вышло: {e}")


trainer = Trainer()


class TrainCfg(BaseModel):
    epochs: int = 80
    model: str = "yolo11n.pt"
    name: str = "parts"
    imgsz: int = 640
    # 0 — подобрать под размер кадра автоматически
    batch: int = 0        # 0 — подобрать под размер кадра автоматически


@app.post("/train/start")
def train_start(c: TrainCfg):
    return trainer.start(c)


@app.get("/train/status")
def train_status():
    return trainer.status()


@app.post("/train/stop")
def train_stop():
    return trainer.stop()


@app.get("/lan")
def lan():
    """Адрес компьютера в сети — чтобы показать в подсказке для телефона."""
    return {"ip": lan_ip(), "port_tls": PORT_TLS, "host": f"{HOSTNAME}.local"}


@app.get("/models")
def models(cam_name: str = Query("A", alias="cam")):
    builtin = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
    own = [p.name for p in MODELS_DIR.glob("*.pt")]
    return {"builtin": builtin, "own": own, "current": cam(cam_name).model_name}


@app.post("/start")
def start(cam_name: str = Query("A", alias="cam")):
    v = cam(cam_name)
    # во время обучения видеокарта занята целиком — распознавание там не поместится
    if trainer.running:
        v.error = "Идёт обучение — видеокарта занята. Дождись конца или останови обучение."
        return v.stats()
    v.start()
    time.sleep(0.8)
    return v.stats()


@app.post("/stop")
def stop(cam_name: str = Query("A", alias="cam")):
    cam(cam_name).stop()
    return {"ok": True}


@app.post("/reset")
def reset(cam_name: str = Query("A", alias="cam")):
    # Сбрасываем только названную камеру. Общего сброса нарочно нет: смысл двух
    # камер в том, чтобы сравнивать их показания, а сравнивать можно только
    # счёт, набранный за один и тот же отрезок времени.
    cam(cam_name).reset()
    return {"ok": True}


@app.post("/config")
def config(c: Config):
    """
    Настройки применяются сразу к ОБЕИМ камерам, без выбора: обе снимают одни
    и те же детали с одного конвейера, так что модель, порог уверенности,
    список классов и цвет всегда должны совпадать. Раздельные остаются только
    счёт и источник запуска/остановки (/start, /stop, /reset, /capture) —
    вот их для сравнения двух камер как раз нельзя схлопывать в одно.
    """
    for v in CAMS.values():
        if c.source is not None:
            # Источник читается один раз, при открытии захвата: какой поток запустить
            # и какую камеру открыть, решается в самом начале. Поменять одну строку
            # мало — служба продолжит читать то, что открыла раньше. Так и выходило:
            # в панели выбран телефон, а кадры идут с веб-камеры; выбрана камера 1,
            # а читается нулевая; вернули веб-камеру после телефона — картинки нет
            # совсем, и ни одной ошибки при этом не показано. Поэтому источник
            # меняем только через полный перезапуск захвата.
            src = c.source.strip()
            if not src:
                # пустое поле — не источник; молча ставить «0» тоже нельзя,
                # иначе человек не поймёт, почему открылась не та камера
                v.error = "Источник не указан. Впиши 0, phone или ссылку rtsp://…"
            elif src != v.source:
                was = v.running
                v.stop()
                v.source = src
                v.error = None      # прежняя жалоба была про прежний источник
                if was:
                    v.start()
        if c.check_color is not None:
            v.check_color = c.check_color
        if c.expect_color is not None:
            want = c.expect_color.strip().lower()
            known = {n for _, _, n in HUES} | {"серый", "белый", "чёрный"}
            v.expect_color = want if want in known else ""
        if c.mirror is not None and not v.is_phone:
            # Когда источник — телефон, зеркалить нельзя: сервер отразил бы кадр
            # ДО распознавания, а телефон рисует рамки поверх своего, неотражённого
            # видео — все рамки уехали бы по горизонтали.
            v.mirror = c.mirror
        if c.conf is not None:
            v.conf = max(0.05, min(0.95, c.conf))
        if c.line_pos is not None:
            v.line_pos = max(0.02, min(0.98, c.line_pos))
        if c.line_orient in ("v", "h"):
            v.line_orient = c.line_orient
        if c.only is not None:
            wanted = [s.strip().lower() for s in c.only.split(",") if s.strip()]
            # разрешаем писать по-русски: переводим обратно в исходные имена модели
            back = {name: k for k, name in RU.items()}
            v.only = {back.get(x, x) for x in wanted}
        if c.model and c.model != v.model_name:
            was = v.running
            v.stop()
            v.load_model(c.model)
            if was:
                v.start()
    return CAMS["A"].stats()


# ── маленький сервер на порту 80: только перенаправляет на https ─────────────
# Набирая «vision.local», браузер сначала стучится по http. Камеру Safari по http
# не даёт, поэтому сразу перекидываем на https — и адрес можно набирать коротко.
redirect_app = FastAPI()


@redirect_app.api_route("/{path:path}", methods=["GET", "POST"])
def to_https(path: str, request: Request):
    host = request.url.hostname or lan_ip()
    q = f"?{request.url.query}" if request.url.query else ""
    return RedirectResponse(f"https://{host}:{PORT_TLS}/{path}{q}", status_code=307)


def serve(target, port: int, tls: bool = False):
    kw = {}
    if tls:
        kw = {"ssl_certfile": str(CERTS / "cert.pem"), "ssl_keyfile": str(CERTS / "key.pem")}
    uvicorn.Server(uvicorn.Config(target, host="0.0.0.0", port=port,
                                  log_level="warning", **kw)).run()


def cert_covers(ip: str) -> bool:
    """
    Покрывает ли сертификат текущий адрес компьютера.
    Роутер выдаёт адрес заново при каждом подключении к сети, и в другом месте
    (например, на заводе) он будет другим. Сертификат, выписанный на старый
    адрес, браузер отвергнет уже всерьёз — поэтому проверяем при каждом старте.
    """
    try:
        from cryptography import x509
        cert = x509.load_pem_x509_certificate((CERTS / "cert.pem").read_bytes())
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        covered = {str(a) for a in san.get_values_for_type(x509.IPAddress)}
        expired = cert.not_valid_after_utc < datetime.now(cert.not_valid_after_utc.tzinfo)
        return ip in covered and not expired
    except Exception:
        return False


def lan_ip() -> str:
    # в контейнере свой адрес — 172.x.x.x, телефону он бесполезен,
    # поэтому настоящий адрес компьютера передаётся через переменную HOST_IP
    if os.environ.get("HOST_IP"):
        return os.environ["HOST_IP"]
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


# ── имя vision.local в сети ──────────────────────────────────────────────────
# Объявляем себя по протоколу mDNS (он же Bonjour). Айфон понимает имена .local
# сам, без единой настройки — в iOS нет файла hosts, и по-другому дать телефону
# короткое имя нельзя.
def start_mdns(ip: str):
    try:
        from zeroconf import IPVersion, ServiceInfo, Zeroconf
    except ImportError:
        return None, []
    try:
        zc = Zeroconf(interfaces=[ip], ip_version=IPVersion.V4Only)
        infos = [
            ServiceInfo(f"_{proto}._tcp.local.", f"Vision._{proto}._tcp.local.",
                        addresses=[socket.inet_aton(ip)], port=port,
                        server=f"{HOSTNAME}.local.", properties={"path": "/"})
            for proto, port in (("https", PORT_TLS), ("http", PORT_HTTP))
        ]
        for i in infos:
            zc.register_service(i)
        return zc, infos
    except OSError:
        return None, []


if __name__ == "__main__":
    ip = lan_ip()
    zc = None

    # сертификата нет или адрес сменился — выписываем заново, сами
    if not (CERTS / "cert.pem").exists() or not cert_covers(ip):
        try:
            import make_cert
            make_cert.main()
        except Exception as e:
            print(f"  Не вышло выписать сертификат: {e}")
    have_cert = (CERTS / "cert.pem").exists() and (CERTS / "key.pem").exists()

    print("=" * 62)
    print("  Vision  ·  распознавание и счёт деталей")
    print("=" * 62)
    print(f"  Видеокарта : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'нет, считаем на процессоре'}")
    print()
    if have_cert:
        zc, _ = start_mdns(ip)
        print(f"  На компьютере :  http://vision.vlx:{PORT}   или  http://127.0.0.1:{PORT}")
        print(f"  На телефоне   :  vision.local:{PORT_TLS}     (или {ip}:{PORT_TLS})")
        print()
        print("  Порты 80/443/8004 заняты MES-системой в Docker, поэтому свои.")
        print()
        print("  Браузер один раз ругнётся на сертификат — это ожидаемо,")
        print("  он свой. Нажми «Подробнее» → «Посетить этот сайт».")
    else:
        print(f"  http://127.0.0.1:{PORT}")
        print("  Для телефона нужен сертификат: запусти make_cert.py")
    for cam_v in CAMS.values():
        saved = cam_v.load_state()
        if saved:
            print(f"  Камера {cam_v.name}: счёт восстановлен из сохранения от {saved}")
    print("=" * 62)
    print("  Чтобы остановить — закрой это окно или нажми Ctrl+C")
    print("=" * 62)

    # окно запущено ярлыком — открываем браузер сами, когда сервер поднимется
    if os.environ.get("VISION_OPEN"):
        threading.Timer(2.5, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()

    try:
        if have_cert:
            threading.Thread(target=serve, args=(app, PORT), daemon=True).start()
            threading.Thread(target=serve, args=(redirect_app, PORT_HTTP), daemon=True).start()
            serve(app, PORT_TLS, True)
        else:
            serve(app, PORT)
    except KeyboardInterrupt:
        pass
    finally:
        vision.stop()
        if zc:
            zc.close()
