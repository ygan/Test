#!/usr/bin/env python3
import argparse
import getpass
import os
import struct
import tarfile
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


MAGIC = b"PYVAULT1"
SALT_LEN = 16
NONCE_LEN = 12

# 如果服务器内存较小，把 2**18 改成 2**16
SCRYPT_N = 2**18
SCRYPT_R = 8
SCRYPT_P = 1


def derive_key(password: str, salt: bytes, n: int) -> bytes:
    kdf = Scrypt(
        salt=salt,
        length=32,
        n=n,
        r=SCRYPT_R,
        p=SCRYPT_P,
    )
    return kdf.derive(password.encode("utf-8"))


def make_tar_gz(input_path: Path, output_path: Path) -> None:
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(input_path, arcname=input_path.name)


def extract_tar_gz(input_path: Path, output_dir: Path) -> None:
    with tarfile.open(input_path, "r:gz") as tar:
        tar.extractall(output_dir)


def encrypt_bytes(data: bytes, password: str) -> bytes:
    salt = os.urandom(SALT_LEN)
    nonce = os.urandom(NONCE_LEN)
    key = derive_key(password, salt, SCRYPT_N)

    ciphertext = AESGCM(key).encrypt(nonce, data, None)

    return (
        MAGIC
        + struct.pack(">I", SCRYPT_N)
        + salt
        + nonce
        + ciphertext
    )


def decrypt_bytes(blob: bytes, password: str) -> bytes:
    if not blob.startswith(MAGIC):
        raise SystemExit("不是本脚本生成的加密文件，或文件已损坏。")

    pos = len(MAGIC)

    n = struct.unpack(">I", blob[pos:pos + 4])[0]
    pos += 4

    salt = blob[pos:pos + SALT_LEN]
    pos += SALT_LEN

    nonce = blob[pos:pos + NONCE_LEN]
    pos += NONCE_LEN

    ciphertext = blob[pos:]
    key = derive_key(password, salt, n)

    try:
        return AESGCM(key).decrypt(nonce, ciphertext, None)
    except Exception:
        raise SystemExit("解密失败：密码错误，或文件被修改/损坏。")


def encrypt_path(input_path: Path, output_path: Path) -> None:
    password = getpass.getpass("Password: ")
    password2 = getpass.getpass("Confirm password: ")

    if password != password2:
        raise SystemExit("两次密码不一致。")

    with tempfile.TemporaryDirectory() as td:
        tmp_tar = Path(td) / "payload.tar.gz"
        make_tar_gz(input_path, tmp_tar)
        data = tmp_tar.read_bytes()

    encrypted = encrypt_bytes(data, password)
    output_path.write_bytes(encrypted)

    print(f"已加密: {output_path}")


def decrypt_path(input_path: Path, output_dir: Path) -> None:
    password = getpass.getpass("Password: ")

    blob = input_path.read_bytes()
    data = decrypt_bytes(blob, password)

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp_tar = Path(td) / "payload.tar.gz"
        tmp_tar.write_bytes(data)
        extract_tar_gz(tmp_tar, output_dir)

    print(f"已解密并解压到: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["enc", "dec"])
    parser.add_argument("input")
    parser.add_argument("output")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.mode == "enc":
        encrypt_path(input_path, output_path)
    else:
        decrypt_path(input_path, output_path)


if __name__ == "__main__":
    main()

# python3 vv.py enc UIE/Archive.zip UIE/Archive.zip.vv
# python3 vv.py dec UIE/Archive.zip.vv UIE/
# pip install --user cryptography