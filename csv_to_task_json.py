#!/usr/bin/env python3
"""Extract a CSV text column into a task dataset for extraction-only reuse."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


def _read_records(
    path: Path, column: str, *, encoding: str, delimiter: str
) -> list[dict[str, str]]:
    records = []
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
                raise ValueError(
                    f"CSV column {column!r} is empty at line {reader.line_num}."
                )
            records.append({"letter": text})

    if not records:
        raise ValueError("The CSV contains no data records.")
    return records


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
    parser.add_argument("--column", required=True, help="Exact name of the text column.")
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
        records = _read_records(
            args.input, args.column, encoding=args.encoding, delimiter=args.delimiter
        )
        _write_records(args.output, records)
    except (csv.Error, LookupError, OSError, ValueError) as exc:
        parser.error(str(exc))

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