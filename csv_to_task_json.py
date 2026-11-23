#!/usr/bin/env python3
"""Extract a CSV text column into a task dataset for extraction-only reuse."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile


def _read_records(
    path: Path, column: str, *, encoding: str, delimiter: str
) -> tuple[list[dict[str, str]], list[int]]:
    records = []
    empty_lines = []
    with path.open(encoding=encoding, newline="") as input_file:
        reader = csv.reader(input_file, delimiter=delimiter, strict=True)
        header = next((row for row in reader if row), [])
        if column not in header:
            raise ValueError(
                f"CSV column {column!r} was not found. Available columns are {header!r}."
            )
        if header.count(column) > 1:
            raise ValueError(
                f"CSV header contains more than one column named {column!r}."
            )
        column_index = header.index(column)

        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(
                    f"CSV record ending at line {reader.line_num} has {len(row)} columns. "
                    f"Expected {len(header)}."
                )
            text = row[column_index]
            if not text.strip():
                empty_lines.append(reader.line_num)
                continue
            records.append({"letter": text})

    if not records:
        raise ValueError(f"CSV column {column!r} has no non-empty text.")
    return records, empty_lines


def _write_records(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as output_file:
            temporary_path = Path(output_file.name)
            json.dump(records, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert one CSV column to a task-style JSON array of letter records.",
        epilog="Run examples/run_final_pipeline.py with --dataset and --no-eval to use the JSON.",
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="CSV file with a header row."
    )
    parser.add_argument(
        "--column", required=True, help="Exact name of the text column. Empty cells are skipped."
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Destination JSON file."
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV encoding. Default is UTF-8 with optional BOM.",
    )
    parser.add_argument(
        "--delimiter", default=",", help="Single-character delimiter. Default is comma."
    )
    args = parser.parse_args()

    if len(args.delimiter) != 1 or args.delimiter in '\r\n"':
        parser.error("--delimiter must be a single valid CSV separator character.")

    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError("Input and output paths must be different.")
        # The default CSV field limit can be too small for long clinical notes.
        csv.field_size_limit(2**31 - 1)
        records, empty_lines = _read_records(
            args.input, args.column, encoding=args.encoding, delimiter=args.delimiter
        )
        _write_records(args.output, records)
    except (csv.Error, LookupError, OSError, ValueError) as exc:
        parser.error(str(exc))

    if empty_lines:
        line_numbers = ", ".join(map(str, empty_lines))
        print(
            f"Skipped {len(empty_lines)} CSV record(s) with empty {args.column!r} cells "
            f"ending on lines {line_numbers}.",
            file=sys.stderr,
        )
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()


# python tools/csv_to_task_json.py \
#     --input notes.csv \
#     --column "letter" \
#     --output generated_outputs/letters.json 

# PYTHONPATH=. python examples/run_final_pipeline.py \
# --pipeline generated_outputs/harness_tasks/epilepsy_classification_single/versions/v001/pipeline.yaml \
# --dataset generated_outputs/letters.json \
# --no-eval

cat > /dataset/singularity/gloo_probe.py <<'PY'
import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])

print(
    f"Rank={rank}, Host={socket.gethostname()}, "
    f"IFACE={os.getenv('GLOO_SOCKET_IFNAME')}",
    flush=True
)

dist.init_process_group(
    backend="gloo",
    timeout=timedelta(seconds=40)
)

x = torch.tensor([float(rank + 1)])
dist.all_reduce(x)
dist.barrier()

print(f"GLOO SUCCESS: rank={rank}, sum={x.item()}", flush=True)

dist.destroy_process_group()
PY

export GLOO_SOCKET_IFNAME=lo

python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    /dataset/singularity/gloo_probe.py