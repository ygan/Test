#!/usr/bin/env python3
"""
Compare two SSL/pip state JSON files and print a short diagnosis.

Usage:
  python compare_tre_ssl_state.py py310_state.json py311_state.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def short(s: str | None, limit: int = 140) -> str:
    if s is None:
        return "None"
    s = str(s).strip().replace("\n", " | ")
    return s if len(s) <= limit else s[:limit - 3] + "..."


def diff_line(name: str, a, b, same_text="相同", diff_text="不同") -> str:
    return f"- {name}: {same_text}" if a == b else f"- {name}: {diff_text}\n  3.10={short(a)}\n  3.11={short(b)}"


def extract_pip_error_text(state: dict) -> str:
    out = get(state, "commands", "pip_install_dry_run", "output", default="") or ""
    if "CERTIFICATE_VERIFY_FAILED" in out:
        return "CERTIFICATE_VERIFY_FAILED"
    if "SSLError" in out:
        return "SSLError"
    if "SSL:" in out:
        return "SSL error"
    if "No matching distribution found" in out:
        return "No matching distribution found"
    if "Could not fetch URL" in out:
        return "Could not fetch URL"
    if out:
        return short(out, 220)
    return "未测试"


def main() -> int:
    if len(sys.argv) != 3:
        print("用法: python compare_tre_ssl_state.py py310_state.json py311_state.json")
        return 2

    a = load(sys.argv[1])  # py310
    b = load(sys.argv[2])  # py311

    lines = []
    lines.append("=== 简短结论 ===")

    ssl_file_a = get(a, "ssl", "file")
    ssl_file_b = get(b, "ssl", "file")
    openssl_a = get(a, "ssl", "OPENSSL_VERSION")
    openssl_b = get(b, "ssl", "OPENSSL_VERSION")
    verify_a = get(a, "ssl", "verify_paths")
    verify_b = get(b, "ssl", "verify_paths")
    pip_ver_a = get(a, "commands", "pip_version", "output")
    pip_ver_b = get(b, "commands", "pip_version", "output")
    env_a = get(a, "env", default={}) or {}
    env_b = get(b, "env", default={}) or {}

    # Diagnosis priority
    if not get(a, "ssl", "import_ok", default=False) or not get(b, "ssl", "import_ok", default=False):
        lines.append("1) 至少一个环境连 ssl 模块导入都不正常，先看 ssl 导入错误。")
    elif ssl_file_a != ssl_file_b:
        lines.append("1) 两个环境导入的 ssl 模块路径不同，优先怀疑模块被覆盖或 Python 安装不同。")
    elif get(a, "_ssl", "import_ok", default=False) != get(b, "_ssl", "import_ok", default=False):
        lines.append("1) 两个环境的 _ssl 扩展状态不同，优先怀疑 3.11 的 Python/OpenSSL 构建或链接有问题。")
    elif verify_a != verify_b:
        lines.append("1) 两个环境的默认 CA 证书路径不同，更像是 3.11 没拿到和 3.10 一样的证书链。")
    elif env_a != env_b:
        lines.append("1) 两个环境的 SSL/PIP 相关环境变量不同，优先检查这些变量。")
    elif pip_ver_a != pip_ver_b:
        lines.append("1) 两个环境 pip 版本不同，可能导致证书处理行为不同。")
    else:
        lines.append("1) 表面配置看起来基本相同；如果 3.11 仍报 SSL 错，更像学校/TRE 侧针对 3.11 模块或证书链的问题。")

    lines.append("")
    lines.append("=== 关键差异 ===")
    lines.append(diff_line("ssl.__file__", ssl_file_a, ssl_file_b))
    lines.append(diff_line("_ssl 导入是否成功", get(a, "_ssl", "import_ok"), get(b, "_ssl", "import_ok")))
    lines.append(diff_line("ssl.OPENSSL_VERSION", openssl_a, openssl_b))
    lines.append(diff_line("默认 verify_paths", verify_a, verify_b))
    lines.append(diff_line("pip --version", pip_ver_a, pip_ver_b))
    lines.append(diff_line("环境变量(env)", env_a, env_b))
    lines.append(diff_line("当前目录有无 ssl.py", get(a, "files", "cwd_ssl_py_exists"), get(b, "files", "cwd_ssl_py_exists")))

    lines.append("")
    lines.append("=== 如果你做了 --test-package 测试 ===")
    lines.append(f"- 3.10 pip 测试结果: {extract_pip_error_text(a)}")
    lines.append(f"- 3.11 pip 测试结果: {extract_pip_error_text(b)}")

    lines.append("")
    lines.append("=== 建议你回给我的最短信息 ===")
    lines.append("把下面几行原样发我即可：")
    lines.append(f"ssl.__file__: 3.10={short(ssl_file_a)} ; 3.11={short(ssl_file_b)}")
    lines.append(f"_ssl import_ok: 3.10={get(a, '_ssl', 'import_ok')} ; 3.11={get(b, '_ssl', 'import_ok')}")
    lines.append(f"verify_paths same?: {verify_a == verify_b}")
    lines.append(f"env same?: {env_a == env_b}")
    lines.append(f"pip error 3.11: {extract_pip_error_text(b)}")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
