# Vision · детекция, трекинг и счёт деталей
# ──────────────────────────────────────────
# Сборка с видеокартой (по умолчанию):
#     docker compose build
# Сборка без видеокарты (образ вчетверо меньше, но распознавание медленнее):
#     docker compose build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
#
# Видеокарта в контейнер пробрасывается — проверено на этой машине:
#     docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
#     -> NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB
# Заводские камеры по RTSP из контейнера тоже видны через VPN — тоже проверено.
#
# Не работает в контейнере только USB-веб-камера ноутбука: Windows не отдаёт
# USB-устройства контейнерам. Телефон работает: он не локальное устройство, а
# шлёт кадры по сети на /ws/camera — достаточно, чтобы был опубликован порт
# 8443 и в сертификате стоял настоящий адрес машины (переменная HOST_IP).

FROM python:3.11-slim

# какие колёса PyTorch ставить: cu124 = с поддержкой видеокарты
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# fonts-dejavu-core — шрифт с кириллицей для подписей на кадре
# (в Linux нет Windows-шрифтов, а запасной шрифт PIL кириллицу не умеет)
# libglib2.0-0 — нужен OpenCV даже в headless-варианте
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Отдельным слоем: torch весит больше всего, пусть кэшируется и не пересобирается
RUN pip install --index-url ${TORCH_INDEX} torch torchvision

# ultralytics подтягивает обычный (GUI) вариант OpenCV, а тому нужны библиотеки
# X11 — в контейнере экрана нет, и import cv2 падает на libxcb.so.1.
# Поэтому сначала ставим всё, потом сносим оба варианта OpenCV
# и ставим headless начисто — так остаётся ровно один, без графики.
RUN pip install \
        ultralytics \
        fastapi \
        "uvicorn[standard]" \
        lap \
        cryptography \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install opencv-python-headless

# zeroconf намеренно НЕ ставим: он объявляет имя vision.local по mDNS, а из
# контейнера это всё равно не работает — своей сети у него нет. В коде импорт
# защищён, без пакета служба просто не объявляет имя.

COPY app.py make_cert.py train.py find_rtsp.py ./
COPY static ./static
# готовая модель внутри образа — чтобы при первом запуске не лезть в интернет
COPY yolo11n.pt ./

# ultralytics хочет писать настройки в /root/.config, куда доступа нет —
# показываем ему сразу папку для временных файлов, чтобы не ругался в журнал
ENV YOLO_CONFIG_DIR=/tmp

# Счётчики, выбранная модель, доступы к камерам и журнал событий пишутся сюда.
# Это том: код живёт в образе и переписывается при каждой сборке, а счёт и
# настройки должны её пережить.
ENV VISION_DATA=/app/data

# 8010 — http и API для этой сети, 8011 — перенаправление на https,
# 8443 — https, здесь работает камера телефона.
# 8090 НЕ занимаем: его публикует zigbee2mqtt из MES.
EXPOSE 8010 8011 8443

# сертификат выписывается при первом запуске, если его ещё нет в томе certs/
CMD ["sh", "-c", "[ -f certs/cert.pem ] || python make_cert.py; exec python app.py"]
