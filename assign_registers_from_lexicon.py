#!/usr/bin/env python3
"""Assign register categories based on a curated lexicon.

This script loads a small manually curated lexicon that lists verbs labelled as
informal (familier/argot) or formal (soutenu) according to authoritative
sources (Larousse, Le Robert, Antidote). It then updates the
`Category_Informal` and `Category_Formal` columns in the verbs CSV.

Usage:
    python assign_registers_from_lexicon.py --input path/to/verbs.csv

By default, a new file named `<input>-assign-formal-cats.csv` is written. Pass
`--output` to override that.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set

DEFAULT_LEXICON = Path(__file__).resolve().parent.parent / "register_lexicon.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign register columns from a curated lexicon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Source CSV")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination CSV. Defaults to <input>-assign-formal-cats.csv",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=DEFAULT_LEXICON,
        help="JSON file containing 'informal' and 'formal' verb lists.",
    )
    parser.add_argument(
        "--preserve",
        action="store_true",
        help="Do not clear existing register values before applying updates.",
    )
    return parser.parse_args()


def ensure_columns(header: List[str], columns: Iterable[str]) -> Dict[str, int]:
    indices: Dict[str, int] = {}
    for column in columns:
        if column not in header:
            header.append(column)
        indices[column] = header.index(column)
    return indices


def load_lexicon(path: Path) -> Dict[str, Set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    informal = {item.strip() for item in data.get("informal", []) if item.strip()}
    formal = {item.strip() for item in data.get("formal", []) if item.strip()}
    overlap = informal & formal
    if overlap:
        raise ValueError(
            "Entries appear in both informal and formal lists: "
            f"{sorted(overlap)!r}"
        )
    return {"informal": informal, "formal": formal}


def assign_registers(
    rows: List[List[str]],
    header_indices: Dict[str, int],
    lexicon: Dict[str, Set[str]],
    preserve: bool,
) -> Dict[str, int]:
    lemma_idx = header_indices.get("lemme")
    if lemma_idx is None:
        raise ValueError("Header must contain a 'lemme' column")

    informal_idx = header_indices["Category_Informal"]
    formal_idx = header_indices["Category_Formal"]

    counts = {"informal": 0, "formal": 0}
    informal = lexicon["informal"]
    formal = lexicon["formal"]

    for row in rows:
        if len(row) <= max(informal_idx, formal_idx, lemma_idx):
            row.extend([""] * (max(informal_idx, formal_idx, lemma_idx) + 1 - len(row)))
        lemma = row[lemma_idx].strip()
        if not lemma:
            continue
        if not preserve:
            row[informal_idx] = ""
            row[formal_idx] = ""
        if lemma in informal:
            row[informal_idx] = "X"
            counts["informal"] += 1
        if lemma in formal:
            row[formal_idx] = "X"
            counts["formal"] += 1
    return counts


def main() -> int:
    args = parse_args()
    if not args.lexicon.exists():
        raise SystemExit(f"Lexicon file not found: {args.lexicon}")

    lexicon = load_lexicon(args.lexicon)

    with args.input.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = list(reader)

    if not rows:
        raise SystemExit(f"CSV is empty: {args.input}")

    header = rows[0]
    header_indices = ensure_columns(
        header,
        ["lemme", "Category_Informal", "Category_Formal"],
    )

    counts = assign_registers(rows[1:], header_indices, lexicon, args.preserve)

    if args.output:
        output_path = args.output
    else:
        suffix = "-assign-formal-cats"
        if args.input.suffix:
            output_path = args.input.with_name(
                f"{args.input.stem}{suffix}{args.input.suffix}"
            )
        else:
            output_path = args.input.with_name(f"{args.input.name}{suffix}")

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)

    print(
        f"Assigned {counts['informal']} informal and {counts['formal']} formal verbs using {args.lexicon.name}."
    )
    print(f"Output written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
