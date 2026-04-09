#!/usr/bin/env python3
"""
Collect compact SSL/pip diagnostics into JSON.

Usage examples:
  python collect_tre_ssl_state.py --out py310_state.json
  python collect_tre_ssl_state.py --out py311_state.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> dict:
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": p.returncode,
            "output": (p.stdout or "").strip(),
        }
    except Exception as e:
        return {
            "cmd": cmd,
            "returncode": None,
            "output": "",
            "error": repr(e),
        }


def module_snapshot(name: str) -> dict:
    snap = {"name": name}
    try:
        mod = importlib.import_module(name)
        snap["import_ok"] = True
        snap["file"] = getattr(mod, "__file__", None)
        snap["attrs_present"] = {}
        for attr in [
            "OPENSSL_VERSION",
            "OPENSSL_VERSION_INFO",
            "HAS_SNI",
            "_DEFAULT_CIPHERS",
            "get_default_verify_paths",
            "create_default_context",
        ]:
            snap["attrs_present"][attr] = hasattr(mod, attr)
            if hasattr(mod, attr) and attr in ("OPENSSL_VERSION", "OPENSSL_VERSION_INFO", "HAS_SNI"):
                try:
                    snap[attr] = getattr(mod, attr)
                except Exception as e:
                    snap[attr] = f"<read failed: {e!r}>"
        if hasattr(mod, "get_default_verify_paths"):
            try:
                vp = mod.get_default_verify_paths()
                snap["verify_paths"] = {
                    "cafile": getattr(vp, "cafile", None),
                    "capath": getattr(vp, "capath", None),
                    "openssl_cafile": getattr(vp, "openssl_cafile", None),
                    "openssl_capath": getattr(vp, "openssl_capath", None),
                }
            except Exception as e:
                snap["verify_paths_error"] = repr(e)
    except Exception as e:
        snap["import_ok"] = False
        snap["error"] = repr(e)
    return snap


def find_pip_configs() -> dict:
    candidates = [
        Path(sys.prefix) / "pip.conf",
        Path.home() / ".pip" / "pip.conf",
        Path.home() / ".config" / "pip" / "pip.conf",
        Path("/etc/pip.conf"),
    ]
    return {str(p): p.exists() for p in candidates}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Path to output JSON file")
    parser.add_argument("--test-package", default="", help="Optional package name for pip dry-run test")
    args = parser.parse_args()

    env_keys = [
        "PIP_CERT",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "PIP_TRUSTED_HOST",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    ]

    data = {
        "sys": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:5]),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "venv_active": sys.prefix != sys.base_prefix,
            "cwd": os.getcwd(),
        },
        "platform": {
            "platform": platform.platform(),
            "python_implementation": platform.python_implementation(),
        },
        "which": {
            "python": shutil.which("python"),
            "python3": shutil.which("python3"),
            "pip": shutil.which("pip"),
            "pip3": shutil.which("pip3"),
            "openssl": shutil.which("openssl"),
        },
        "env": {k: os.environ.get(k) for k in env_keys if os.environ.get(k) is not None},
        "files": {
            "cwd_ssl_py_exists": (Path.cwd() / "ssl.py").exists(),
            "cwd_ssl_pyc_exists": any((Path.cwd() / "__pycache__").glob("ssl.*.pyc")) if (Path.cwd() / "__pycache__").exists() else False,
            "pip_configs_exist": find_pip_configs(),
        },
        "ssl": module_snapshot("ssl"),
        "_ssl": module_snapshot("_ssl"),
        "commands": {
            "pip_version": run_cmd([sys.executable, "-m", "pip", "--version"]),
            "pip_config_list_v": run_cmd([sys.executable, "-m", "pip", "config", "list", "-v"]),
            "pip_config_debug": run_cmd([sys.executable, "-m", "pip", "config", "debug"]),
            "pip_debug_v": run_cmd([sys.executable, "-m", "pip", "debug", "-v"]),
        },
    }

    if data["which"]["openssl"]:
        data["commands"]["openssl_version"] = run_cmd([data["which"]["openssl"], "version", "-a"])

    if args.test_package:
        data["commands"]["pip_install_dry_run"] = run_cmd(
            [sys.executable, "-m", "pip", "install", args.test_package, "-v", "--dry-run"]
        )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
