"""Export operational Gan results in LLaMA Factory prediction format.

Each successful input row becomes one JSONL row containing only the public Gan
answer under LLaMA Factory's lowercase ``predict`` key.

Usage:

    python export_llamafactory_predictions.py \
        --input predictions.jsonl \
        --output generated_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[tuple[int, Any]]:
    rows: list[tuple[int, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rows.append((line_number, json.loads(line)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
    if not rows:
        raise ValueError(f"{path}: no prediction rows found")
    return rows


def export_predictions(input_path: Path, output_path: Path) -> int:
    exported: list[dict[str, str]] = []
    for line_number, row in _read_jsonl(input_path):
        if not isinstance(row, Mapping):
            raise ValueError(f"{input_path}:{line_number}: row must be a JSON object")
        if row.get("status") != "ok":
            raise ValueError(
                f"{input_path}:{line_number}: cannot export row with status "
                f"{row.get('status')!r}"
            )
        prediction = row.get("prediction")
        if not isinstance(prediction, Mapping):
            raise ValueError(f"{input_path}:{line_number}: missing prediction object")
        final_answer = prediction.get("seizure_frequency")
        if not isinstance(final_answer, str) or not final_answer.strip():
            raise ValueError(
                f"{input_path}:{line_number}: missing prediction.seizure_frequency"
            )
        exported.append({"predict": final_answer})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in exported:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    return len(exported)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Gan output JSONL")
    parser.add_argument(
        "--output", type=Path, required=True, help="LLaMA Factory-style output JSONL"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists; pass --overwrite to replace it: {args.output}")
    try:
        count = export_predictions(args.input, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"rows": count, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
