# -*- coding: utf-8 -*-
"""
Создание самоподписанного сертификата для HTTPS
────────────────────────────────────────────────
Зачем: Safari на айфоне не даёт странице доступ к камере по обычному http://.
Только https://. Настоящий сертификат для домашнего IP взять негде, поэтому
выписываем свой. Браузер один раз ругнётся — надо будет разрешить вручную.

Запуск:
    .venv\\Scripts\\python.exe make_cert.py

Сертификат кладётся в certs/ и подхватывается app.py автоматически.
Если сменился IP компьютера — просто запусти скрипт заново.
"""
import datetime
import ipaddress
import os
import socket
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

BASE = Path(__file__).parent
CERTS = BASE / "certs"


def local_ips() -> list[str]:
    """Все адреса компьютера в сети — сертификат должен покрывать их все."""
    ips = {"127.0.0.1"}
    # в контейнере видны только внутренние адреса, поэтому настоящий адрес
    # компьютера передаётся снаружи: HOST_IP=192.168.1.6 (или CERT_IPS через запятую)
    for var in ("HOST_IP", "CERT_IPS"):
        for extra in os.environ.get(var, "").split(","):
            extra = extra.strip()
            if extra:
                ips.add(extra)
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    # адрес, через который машина ходит в интернет — обычно он и нужен
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
    except OSError:
        pass
    finally:
        s.close()
    return sorted(ips)


def main():
    CERTS.mkdir(exist_ok=True)
    ips = local_ips()
    print("Сертификат будет действовать для адресов:")
    for ip in ips:
        print("   ", ip)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Vision Local"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Vision"),
    ])

    # SAN обязателен: современные браузеры игнорируют CommonName,
    # а iOS ещё и требует, чтобы IP лежал именно в IPAddress, а не в DNSName
    san = [x509.DNSName(n) for n in ("localhost", "vision.local", "vision.vlx")]
    for ip in ips:
        san.append(x509.IPAddress(ipaddress.ip_address(ip)))
    print("и для имён: localhost, vision.local, vision.vlx")

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=825))   # предел, который принимает iOS
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    (CERTS / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (CERTS / "key.pem").write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    print(f"\nГотово: {CERTS}")
    print("Теперь запусти app.py — он сам поднимет https на порту 8443.")


if __name__ == "__main__":
    main()
