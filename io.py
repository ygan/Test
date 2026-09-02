"""Strict JSONL input and atomic output helpers."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InputNote:
    note_id: str
    text: str


def read_notes(path: Path) -> list[InputNote]:
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError(f"{path}: no input notes found")

    rows: list[tuple[int, Any]]
    try:
        document = json.loads(content)
    except json.JSONDecodeError:
        rows = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append((line_number, json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
    else:
        if isinstance(document, list):
            rows = list(enumerate(document, start=1))
        else:
            rows = [(1, document)]

    notes: list[InputNote] = []
    seen: set[str] = set()
    for record_number, row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{record_number}: each row must be a JSON object")
        note_id = str(row.get("id", "")).strip() or f"note_{record_number}"
        text = str(row.get("text", "")).strip() or str(row.get("test", "")).strip()
        if not text:
            raise ValueError(
                f"{path}:{record_number}: non-empty 'text' or 'test' is required"
            )
        if note_id in seen:
            raise ValueError(f"{path}:{record_number}: duplicate id {note_id!r}")
        seen.add(note_id)
        notes.append(InputNote(note_id=note_id, text=text))
    if not notes:
        raise ValueError(f"{path}: no input notes found")
    return notes


def write_jsonl_atomic(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
