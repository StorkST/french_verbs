#!/usr/bin/env python3
"""
Analyse French Wiktionary entries to infer stylistic registers.

The script scans each lemma in the verbs CSV, fetches the corresponding page
from fr.wiktionary.org and looks for templates or labels that indicate whether
the verb is typically informal/familiar or formal/sustained. When a match is
detected, the script writes "X" into the `Category_Informal` or
`Category_Formal` columns of the CSV (adding the columns if necessary).

Usage:
    python assign_registers_from_wiktionary.py \
        --input /path/to/Verbes_en_français-2025-11-09.csv \
        --output /path/to/Verbes_en_français-annotated.csv

Key options let you control rate limiting, retries and whether to overwrite
existing values. See `--help` for details.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests


WIKTIONARY_ENDPOINT = "https://fr.wiktionary.org/w/api.php"

# Map CSV column names to regex patterns that signal the register.
# Patterns are matched on raw wikitext (lower-cased). They intentionally
# cast a wide net to catch the most common template variants.
REGISTER_PATTERNS: Dict[str, Iterable[re.Pattern[str]]] = {
    "Category_Informal": [
        re.compile(pattern, re.IGNORECASE | re.DOTALL)
        for pattern in (
            r"\{\{\s*(?:fr-)?(?:familier|très\s*familier|informel)\b",
            r"\{\{\s*(?:fr-)?populaire\b",
            r"\{\{\s*(?:fr-)?argot(?:ique)?\b",
            r"\{\{\s*(?:label|lb|qualifier|qual|usage|registre)\|[^}]*"
            r"(?:familier|informel|populaire|argotique)\b",
        )
    ],
    "Category_Formal": [
        re.compile(pattern, re.IGNORECASE | re.DOTALL)
        for pattern in (
            r"\{\{\s*(?:fr-)?(?:soutenu|très\s*soutenu)\b",
            r"\{\{\s*(?:fr-)?littéraire\b",
            r"\{\{\s*(?:label|lb|qualifier|qual|usage|registre)\|[^}]*"
            r"(?:soutenu|littéraire|soigné)\b",
        )
    ],
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate register columns using French Wiktionary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Source CSV containing a `lemme` column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination CSV. Defaults to in-place overwrite of --input.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Delay (in seconds) between API requests to avoid rate limits.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts per lemma on HTTP errors.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite register columns even if they already contain data.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional directory to cache raw wikitext responses.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Stop after processing this many lemmas (for testing).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def ensure_columns(header: List[str], columns: Iterable[str]) -> Dict[str, int]:
    """Ensure all requested columns exist. Append them if missing."""
    indices: Dict[str, int] = {}
    for column in columns:
        if column not in header:
            header.append(column)
        indices[column] = header.index(column)
    return indices


def slugify(title: str) -> str:
    """Return a filesystem-friendly cache key for a Wiktionary page title."""
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()
    return f"{digest}.txt"


def load_wikitext(
    title: str,
    session: requests.Session,
    *,
    delay: float,
    retries: int,
    cache_dir: Optional[Path] = None,
) -> Optional[str]:
    """Fetch raw wikitext for a page, optionally using a cache."""
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / slugify(title)
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")
    else:
        cache_path = None

    params = {
        "action": "parse",
        "page": title,
        "prop": "wikitext",
        "format": "json",
    }

    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                WIKTIONARY_ENDPOINT,
                params=params,
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            logging.warning("Request failed for %s (attempt %s/%s): %s", title, attempt, retries, exc)
            time.sleep(delay)
            continue

        if "error" in data:
            if data["error"].get("code") == "missingtitle":
                logging.info("No Wiktionary entry found for %s", title)
                return None
            logging.warning("API error for %s: %s", title, data["error"])
            time.sleep(delay)
            continue

        try:
            wikitext = data["parse"]["wikitext"]["*"]
        except KeyError:
            logging.warning("Unexpected response shape for %s: %s", title, data)
            time.sleep(delay)
            continue

        if cache_path:
            cache_path.write_text(wikitext, encoding="utf-8")
        time.sleep(delay)
        return wikitext

    logging.error("Giving up on %s after %s attempts", title, retries)
    return None


def detect_registers(wikitext: str) -> Dict[str, str]:
    """Return a mapping of register columns to 'X' or '' based on wikitext."""
    result = {column: "" for column in REGISTER_PATTERNS}
    if not wikitext:
        return result

    for column, patterns in REGISTER_PATTERNS.items():
        if any(pattern.search(wikitext) for pattern in patterns):
            result[column] = "X"
    return result


def update_csv(
    input_path: Path,
    output_path: Optional[Path],
    *,
    sleep_seconds: float,
    max_retries: int,
    overwrite: bool,
    cache_dir: Optional[Path],
    limit: Optional[int],
) -> None:
    with input_path.open("r", newline="", encoding="utf-8") as fh:
        reader = list(csv.reader(fh))

    if not reader:
        logging.error("Input CSV %s is empty.", input_path)
        return

    header = reader[0]
    try:
        lemma_index = header.index("lemme")
    except ValueError as exc:
        raise SystemExit("The CSV must contain a 'lemme' column.") from exc

    column_indices = ensure_columns(header, REGISTER_PATTERNS.keys())

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "RegisterCategoriser/1.0 "
                "(https://github.com/JobsAround; contact: data-team@jobsaround.example)"
            )
        }
    )

    processed = 0

    for row_num, row in enumerate(reader[1:], start=2):
        lemma = row[lemma_index].strip()
        if not lemma:
            logging.debug("Skipping empty lemma at row %s", row_num)
            continue

        if limit is not None and processed >= limit:
            logging.info("Reached processing limit of %s entries.", limit)
            break

        existing_values = {col: row[idx].strip() for col, idx in column_indices.items()}
        if not overwrite and all(existing_values[col] for col in column_indices):
            logging.debug("Skipping %s (already populated).", lemma)
            continue

        logging.info("Processing lemma '%s' (row %s)", lemma, row_num)
        title = lemma.replace("’", "'").replace(" ", "_")
        wikitext = load_wikitext(
            title,
            session,
            delay=sleep_seconds,
            retries=max_retries,
            cache_dir=cache_dir,
        )
        registers = detect_registers(wikitext or "")
        for column, value in registers.items():
            idx = column_indices[column]
            if value == "X" or overwrite:
                row[idx] = value

        processed += 1

    destination = output_path or input_path
    with destination.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(reader)

    logging.info(
        "Finished updating %s entries. Output written to %s.",
        processed,
        destination,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_path = args.output if args.output else None

    try:
        update_csv(
            args.input,
            output_path,
            sleep_seconds=args.sleep,
            max_retries=args.max_retries,
            overwrite=args.overwrite,
            cache_dir=args.cache_dir,
            limit=args.limit,
        )
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001 - surface full error to CLI
        logging.exception("Unexpected failure: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
