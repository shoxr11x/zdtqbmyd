"""
Подбор ссылки RTSP по адресу камеры.

У разных марок путь к потоку разный, и угадывать его по названию — гиблое
дело: одна и та же фирма меняет путь от прошивки к прошивке. Проще перебрать
известные варианты и посмотреть, какой ответит. Скрипт открывает каждый,
читает несколько кадров и показывает настоящее разрешение и частоту.

Запуск (пароль в кавычках, если в нём есть спецсимволы):

    python find_rtsp.py 192.168.1.64 admin "пароль"

Можно сразу две камеры:

    python find_rtsp.py 192.168.1.64 192.168.1.65 admin "пароль"

Пароль нигде не сохраняется — он живёт только в этом запуске.
"""
import os
import socket
import sys
import time

# TCP и таймаут — те же, что в самой программе: проверять надо в тех условиях,
# в которых потом будет работать
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|stimeout;5000000|max_delay;500000"
)
import cv2

# Пути, которые встречаются чаще всего. Сначала «второй поток» (substream):
# он обычно 640-1080 и его хватает, а главный часто 4K — на нём одно только
# разжатие кадра стоит около 50 мс, две такие камеры машина не потянет.
PATHS = [
    # Hikvision и всё, что сделано на их прошивке
    ("/Streaming/Channels/102", "Hikvision, второй поток"),
    ("/Streaming/Channels/101", "Hikvision, главный поток"),
    ("/h264/ch1/sub/av_stream", "Hikvision старый, второй"),
    ("/h264/ch1/main/av_stream", "Hikvision старый, главный"),
    # Dahua, Imou и родня
    ("/cam/realmonitor?channel=1&subtype=1", "Dahua, второй поток"),
    ("/cam/realmonitor?channel=1&subtype=0", "Dahua, главный поток"),
    # частые универсальные
    ("/stream2", "универсальный, второй"),
    ("/stream1", "универсальный, главный"),
    ("/live/ch1", "универсальный live"),
    ("/onvif1", "ONVIF, второй"),
    ("/onvif0", "ONVIF, главный"),
    ("/11", "TP-Link/Tapo, главный"),
    ("/12", "TP-Link/Tapo, второй"),
    ("/video1", "универсальный video"),
    ("", "без пути (порт отдаёт поток сам)"),
]


def port_open(ip: str, port: int = 554, timeout: float = 2.0) -> bool:
    """Живой ли вообще RTSP-порт. Если нет — перебирать пути бессмысленно."""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def describe(ip: str, path: str, port: int = 554, timeout: float = 3.0):
    """
    Спросить камеру про поток БЕЗ пароля и посмотреть на код ответа.

    Пароль для этого не нужен, и это важно: путь можно найти, ничего секретного
    никуда не вводя. Камера отвечает:
      401 — путь есть, просто нужен логин и пароль  <- то, что мы ищем
      200 — путь есть и открыт вообще без пароля
      404 — такого пути у неё нет
    Заодно в ответе на 401 приходит realm, а по нему часто видно марку.
    """
    req = (f"DESCRIBE rtsp://{ip}:{port}{path} RTSP/1.0\r\n"
           f"CSeq: 1\r\nUser-Agent: vision-probe\r\nAccept: application/sdp\r\n\r\n")
    try:
        with socket.create_connection((ip, port), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall(req.encode())
            data = b""
            while b"\r\n\r\n" not in data and len(data) < 4096:
                chunk = s.recv(1024)
                if not chunk:
                    break
                data += chunk
    except OSError:
        return None, ""
    text = data.decode("latin-1", "replace")
    first = text.split("\r\n", 1)[0]
    code = 0
    parts = first.split()
    if len(parts) >= 2 and parts[1].isdigit():
        code = int(parts[1])
    realm = ""
    for line in text.split("\r\n"):
        if line.lower().startswith("www-authenticate"):
            realm = line.split("realm=", 1)[-1].strip('" ').split('"')[0] if "realm=" in line else ""
    return code, realm


def scan_no_password(ip: str):
    """Найти путь к потоку, не зная пароля."""
    print(f"\n=== камера {ip} — ищем путь без пароля ===")
    if not port_open(ip):
        print("  Порт 554 не отвечает. Либо камера выключена, либо нет связи")
        print("  с заводской сетью — проверь VPN.")
        return []
    good = []
    for path, note in PATHS:
        code, realm = describe(ip, path)
        if code in (200, 401):
            mark = "нужен пароль" if code == 401 else "открыт без пароля"
            extra = f", камера представляется «{realm}»" if realm else ""
            print(f"  [+] {note:<32} {code} — {mark}{extra}")
            good.append((path, note, code, realm))
        elif code:
            print(f"  [-] {note:<32} {code}")
        else:
            print(f"  [?] {note:<32} нет ответа")
    return good


def try_url(url: str, seconds: float = 3.0):
    """Открыть и померить. Возвращает (ширина, высота, к/с) или None."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None
    h, w = frame.shape[:2]
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    fps = n / (time.perf_counter() - t0)
    cap.release()
    return w, h, fps


def scan(ip: str, user: str, pwd: str):
    print(f"\n=== камера {ip} ===")
    if not port_open(ip):
        print("  Порт 554 закрыт. Проверь: включена ли камера, тот ли адрес,")
        print("  и подключён ли VPN (без него заводская сеть недоступна).")
        return []

    # экранируем спецсимволы в пароле, иначе ссылка развалится
    from urllib.parse import quote
    auth = f"{quote(user, safe='')}:{quote(pwd, safe='')}@" if user else ""
    found = []
    for path, note in PATHS:
        url = f"rtsp://{auth}{ip}:554{path}"
        r = try_url(url)
        shown = url.replace(f":{quote(pwd, safe='')}@", ":***@") if pwd else url
        if r:
            w, h, fps = r
            mp = w * h / 1e6
            verdict = ("хорошо" if mp <= 2.5 else "тяжёлый, лучше второй поток")
            print(f"  [+] {note:<32} {w}x{h} {fps:>5.1f} к/с  {mp:.1f} Мп — {verdict}")
            print(f"      {shown}")
            found.append((mp, url, shown, w, h, fps))
        else:
            print(f"  [-] {note}")
    return found


def main():
    args = sys.argv[1:]
    ips = [a for a in args if a.count(".") == 3]
    rest = [a for a in args if a not in ips]
    if not ips:
        print(__doc__)
        return
    user = rest[0] if rest else ""
    pwd = rest[1] if len(rest) > 1 else ""

    # Без пароля — только ищем путь. Это безопасно и этого достаточно, чтобы
    # понять, что вписывать: пароль потом добавишь сам в поле панели.
    if not pwd:
        paths = {ip: scan_no_password(ip) for ip in ips}
        print("\n" + "=" * 64)
        print("ЧТО ВПИСАТЬ В ПАНЕЛЬ")
        print("=" * 64)
        for ip, found in paths.items():
            if not found:
                print(f"  {ip}: путь не определился.")
                continue
            path, note, code, _ = found[0]      # первый в списке — самый лёгкий поток
            if code == 401:
                print(f"  {ip}  ->  rtsp://ЛОГИН:ПАРОЛЬ@{ip}:554{path}")
                print(f"          ({note}; логин и пароль впиши вместо заглавных)")
            else:
                print(f"  {ip}  ->  rtsp://{ip}:554{path}   ({note}, пароль не нужен)")
        print("\n  Панель: «Чей источник настраиваем» -> Камера A -> вставить")
        print("  первую ссылку -> «Запустить». Потом Камера B -> вторая ссылка.")
        print("\n  Проверить со своим паролем (он никуда не сохраняется):")
        print(f"    python find_rtsp.py {' '.join(ips)} логин \"пароль\"")
        return

    all_found = {}
    for ip in ips:
        all_found[ip] = scan(ip, user, pwd)

    print("\n" + "=" * 64)
    print("ЧТО ВПИСАТЬ В ПАНЕЛЬ")
    print("=" * 64)
    any_ok = False
    for ip, found in all_found.items():
        if not found:
            print(f"  {ip}: ничего не ответило.")
            continue
        any_ok = True
        # берём самый лёгкий из рабочих: он и есть второй поток
        found.sort()
        mp, url, shown, w, h, fps = found[0]
        print(f"  {ip}  ->  {url}")
        print(f"          {w}x{h}, {fps:.1f} к/с, {mp:.1f} Мп")
    if any_ok:
        print("\n  В панели: «Чей источник настраиваем» -> Камера A -> вписать")
        print("  первую ссылку -> «Запустить». Потом Камера B -> вторая ссылка")
        print("  -> «Запустить». Когда обе покажут «идут кадры» — всё готово.")
    if not any_ok:
        print("\n  Если порт открыт, а путь не подошёл — посмотри точную ссылку")
        print("  в веб-интерфейсе самой камеры, раздел «RTSP» или «Поток».")


if __name__ == "__main__":
    main()
