"""红星光 CSL S1 一键启动脚本：生成自签 HTTPS 证书（若缺失）→ 启动 HTTPS 服务。

平板经 https://192.168.1.8:8000 访问（浏览器点"继续访问"接受自签证书即可用摄像头）。
本机调试加 --no-https（明文 HTTP，仅限 127.0.0.1）。
"""

from __future__ import annotations

import ipaddress
import socket
import sys
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

ROOT = Path(__file__).resolve().parent
CERT_DIR = ROOT / "certs"
CERT, KEY = CERT_DIR / "cert.pem", CERT_DIR / "key.pem"
DAYS = 365


def _lan_ip() -> str | None:
    """取本机局域网 IPv4（WLAN 优先）。拿不到时回退 127.0.0.1。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = ipaddress.ip_address(info[4][0])
            if not ip.is_loopback:
                return str(ip)
        return None
    finally:
        s.close()


def ensure_cert() -> Path:
    """证书缺失时生成 365 天自签证书（SAN 含局域网 IP / 127.0.0.1 / localhost）。"""
    if CERT.is_file() and KEY.is_file():
        return CERT

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    lan = _lan_ip() or "127.0.0.1"
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, lan)])
    san = x509.SubjectAlternativeName(
        [x509.IPAddress(ipaddress.ip_address(lan))]
        + [x509.IPAddress(ipaddress.ip_address("127.0.0.1")), x509.DNSName("localhost")]
    )
    now = x509.datetime_now()
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now.replace(year=now.year + DAYS))
        .add_extension(san, critical=False)
        .sign(key, hashes.SHA256())
    )
    KEY.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    CERT.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return CERT


if __name__ == "__main__":
    cert = ensure_cert()
    from backend.main import main

    print(f"[start] HTTPS 证书就绪: {cert}（SAN 含局域网 IP，有效期 {DAYS} 天）")
    if "--no-https" not in sys.argv:
        lan = _lan_ip() or "127.0.0.1"
        print(f"[start] 平板浏览器访问: https://{lan}:8000 （首次需点\"继续访问\"接受自签证书）")
    main()
