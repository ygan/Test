#!/usr/bin/env python3
"""
TRE / HPC Python SSL & pip diagnostics

Usage:
    python tre_ssl_diagnose.py
    python tre_ssl_diagnose.py --test-pip pip
    python tre_ssl_diagnose.py --save report.txt
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path


def line(char: str = "=", width: int = 78) -> str:
    return char * width


def section(title: str) -> str:
    return f"\n{line()}\n{title}\n{line()}"


def safe_repr(value) -> str:
    try:
        return repr(value)
    except Exception as e:
        return f"<repr failed: {e!r}>"


def run_cmd(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        header = f"$ {' '.join(cmd)}\n[exit code: {proc.returncode}]"
        body = proc.stdout.rstrip() if proc.stdout else "<no output>"
        return f"{header}\n{body}"
    except Exception as e:
        return f"$ {' '.join(cmd)}\n[execution failed] {e!r}"


def module_info(name: str) -> str:
    out = [section(f"MODULE: {name}")]
    try:
        mod = importlib.import_module(name)
        out.append(f"module repr: {mod!r}")
        out.append(f"__file__: {getattr(mod, '__file__', '<no __file__>')}")
        attrs = dir(mod)
        out.append(f"attribute count: {len(attrs)}")
        interesting = [
            x for x in attrs
            if (
                "SSL" in x
                or "verify" in x.lower()
                or "cert" in x.lower()
                or "openssl" in x.lower()
            )
        ]
        out.append("interesting attrs:")
        if interesting:
            for x in interesting[:80]:
                out.append(f"  - {x}")
        else:
            out.append("  <none>")
        for attr in [
            "OPENSSL_VERSION",
            "OPENSSL_VERSION_INFO",
            "HAS_SNI",
            "_DEFAULT_CIPHERS",
        ]:
            out.append(f"hasattr({attr}): {hasattr(mod, attr)}")
            if hasattr(mod, attr):
                try:
                    out.append(f"  value: {safe_repr(getattr(mod, attr))}")
                except Exception as e:
                    out.append(f"  value read failed: {e!r}")

        if hasattr(mod, "get_default_verify_paths"):
            try:
                out.append(
                    f"get_default_verify_paths(): "
                    f"{safe_repr(mod.get_default_verify_paths())}"
                )
            except Exception as e:
                out.append(f"get_default_verify_paths() failed: {e!r}")

        if hasattr(mod, "create_default_context"):
            try:
                ctx = mod.create_default_context()
                out.append(f"create_default_context(): OK -> {ctx!r}")
            except Exception as e:
                out.append(f"create_default_context() failed: {e!r}")

    except Exception as e:
        out.append(f"IMPORT FAILED: {e!r}")

    return "\n".join(out)


def env_info() -> str:
    keys = sorted(
        k for k in os.environ
        if k.startswith(("PIP", "SSL", "REQUESTS", "CURL", "PYTHON"))
    )
    out = [section("ENVIRONMENT VARIABLES")]
    if not keys:
        out.append("<no matching env vars>")
    else:
        for k in keys:
            out.append(f"{k}={os.environ.get(k)}")
    return "\n".join(out)


def python_info() -> str:
    out = [section("PYTHON / PLATFORM INFO")]
    out.append(f"timestamp: {datetime.now().isoformat()}")
    out.append(f"sys.executable: {sys.executable}")
    out.append(f"sys.version: {sys.version}")
    out.append(f"sys.prefix: {sys.prefix}")
    out.append(f"sys.base_prefix: {sys.base_prefix}")
    out.append(f"venv active: {sys.prefix != sys.base_prefix}")
    out.append(f"platform: {platform.platform()}")
    out.append(f"python implementation: {platform.python_implementation()}")
    out.append(f"cwd: {os.getcwd()}")
    out.append(f"PATH: {os.environ.get('PATH')}")
    return "\n".join(out)


def path_checks() -> str:
    out = [section("COMMON FILE CHECKS")]
    candidates = [
        Path.cwd() / "ssl.py",
        Path.cwd() / "__pycache__",
        Path(sys.prefix) / "pip.conf",
        Path.home() / ".pip" / "pip.conf",
        Path.home() / ".config" / "pip" / "pip.conf",
    ]
    for p in candidates:
        out.append(f"{p}: {'EXISTS' if p.exists() else 'missing'}")
    return "\n".join(out)


def pip_info() -> str:
    out = [section("PIP COMMANDS")]
    pip_module = [sys.executable, "-m", "pip"]

    commands = [
        pip_module + ["--version"],
        pip_module + ["config", "debug"],
        pip_module + ["config", "list", "-v"],
        pip_module + ["debug", "-v"],
    ]

    for cmd in commands:
        out.append(run_cmd(cmd))
        out.append(line("-"))

    return "\n".join(out)


def openssl_info() -> str:
    out = [section("OPENSSL / CERT TOOLS")]
    openssl = shutil.which("openssl")
    if openssl:
        out.append(run_cmd([openssl, "version", "-a"]))
    else:
        out.append("openssl command not found in PATH")

    for cmd_name in ["python", "python3", "pip", "pip3"]:
        found = shutil.which(cmd_name)
        out.append(f"which {cmd_name}: {found}")
    return "\n".join(out)


def optional_pip_test(package: str) -> str:
    out = [section(f"OPTIONAL PIP TEST: {package}")]
    cmd = [sys.executable, "-m", "pip", "install", package, "-v", "--dry-run"]
    out.append(run_cmd(cmd))
    return "\n".join(out)


def build_report(test_pip_package: str | None) -> str:
    parts = [
        python_info(),
        env_info(),
        path_checks(),
        module_info("ssl"),
        module_info("_ssl"),
        openssl_info(),
        pip_info(),
    ]
    if test_pip_package:
        parts.append(optional_pip_test(test_pip_package))
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose Python SSL / pip issues on HPC or restricted environments."
    )
    parser.add_argument(
        "--test-pip",
        metavar="PACKAGE",
        help="Optionally run: python -m pip install PACKAGE -v --dry-run",
    )
    parser.add_argument(
        "--save",
        metavar="FILE",
        help="Save output to a text file as well as printing it",
    )
    args = parser.parse_args()

    report = build_report(args.test_pip)

    print(report)

    if args.save:
        path = Path(args.save)
        path.write_text(report, encoding="utf-8")
        print(section("SAVED"))
        print(f"Report written to: {path.resolve()}")

    print(section("HOW TO USE"))
    print(textwrap.dedent(f"""\
        1. In your Python 3.10 environment:
           python {Path(__file__).name} --save py310_report.txt

        2. In your Python 3.11 environment:
           python {Path(__file__).name} --save py311_report.txt

        3. Compare:
           - module path for ssl / _ssl
           - whether OPENSSL_VERSION exists
           - get_default_verify_paths()
           - pip config debug / list -v
           - any SSL_* / PIP_* environment variables

        4. If you want, also test pip resolution without actually installing:
           python {Path(__file__).name} --test-pip pip --save report_with_pip_test.txt
        """))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
