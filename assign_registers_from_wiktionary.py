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
        --input /path/to/Verbes_en_français-2025-11-09.csv

By default the script writes the annotated data to a sibling file whose name
ends with ``-assign-formal-regs.csv``. Override with ``--output`` if you want a
different destination. Key options let you control rate limiting, concurrency,
retries and whether to overwrite existing values. See ``--help`` for details.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import logging
import re
import sys
import time
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests


WIKTIONARY_ENDPOINT = "https://fr.wiktionary.org/w/api.php"
DEFAULT_USER_AGENT = (
    "RegisterCategoriser/1.1 "
    "(https://github.com/JobsAround; contact: data-team@jobsaround.example)"
)

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


class RateLimiter:
    """Thread-safe limiter that enforces a minimum delay between API calls."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = max(min_interval, 0.0)
        self._lock = threading.Lock()
        self._next_available: float = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next_available:
                    self._next_available = now + self.min_interval
                    return
                wait_time = self._next_available - now
            # Sleep outside the lock to allow other threads to progress.
            time.sleep(wait_time)


class ProgressTracker:
    """Light-weight stderr progress indicator for long-running jobs."""

    def __init__(self, total: int, interval: int = 50) -> None:
        self.total = total
        self.interval = max(interval, 1)
        self.completed = 0
        self._lock = threading.Lock()
        self._printed = False

    def advance(self) -> None:
        with self._lock:
            self.completed += 1
            if self.completed % self.interval and self.completed != self.total:
                return
            percent = (self.completed / self.total * 100) if self.total else 0.0
            sys.stderr.write(
                f"\rProgress: {self.completed}/{self.total} "
                f"({percent:5.1f}%)"
            )
            sys.stderr.flush()
            self._printed = True

    def finish(self) -> None:
        with self._lock:
            if self._printed:
                sys.stderr.write("\n")
                sys.stderr.flush()
                self._printed = False


_THREAD_LOCAL = threading.local()


def get_session(user_agent: str) -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": user_agent})
        _THREAD_LOCAL.session = session
    return session


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
        help="Destination CSV. Defaults to <input>-assign-formal-cats.csv.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers fetching Wiktionary pages.",
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
        "--progress-interval",
        type=int,
        default=50,
        help="Emit a progress update after this many completed lemmas.",
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
    rate_limiter: Optional[RateLimiter] = None,
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
            if rate_limiter:
                rate_limiter.wait()
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

    if result["Category_Informal"] and result["Category_Formal"]:
        logging.debug(
            "Both informal and formal markers detected; clearing for neutrality."
        )
        for column in result:
            result[column] = ""
    return result


def update_csv(
    input_path: Path,
    output_path: Optional[Path],
    *,
    sleep_seconds: float,
    workers: int,
    progress_interval: int,
    max_retries: int,
    overwrite: bool,
    cache_dir: Optional[Path],
    limit: Optional[int],
) -> int:
    with input_path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))

    if not rows:
        logging.error("Input CSV %s is empty.", input_path)
        return 0

    header = rows[0]
    try:
        lemma_index = header.index("lemme")
    except ValueError as exc:
        raise SystemExit("The CSV must contain a 'lemme' column.") from exc

    column_indices = ensure_columns(header, REGISTER_PATTERNS.keys())

    jobs: List[Tuple[int, str]] = []
    total_rows = len(rows) - 1
    max_jobs = min(limit, total_rows) if limit is not None else None
    limit_triggered = False

    for row_idx, row in enumerate(rows[1:], start=1):
        if max_jobs is not None and len(jobs) >= max_jobs:
            limit_triggered = True
            break
        if len(row) < len(header):
            row.extend([""] * (len(header) - len(row)))
        lemma = row[lemma_index].strip()
        if not lemma:
            logging.debug("Skipping empty lemma at row %s", row_idx + 1)
            continue

        existing_values = {col: row[idx].strip() for col, idx in column_indices.items()}
        if not overwrite and all(existing_values[col] for col in column_indices):
            logging.debug("Skipping %s (already populated).", lemma)
            continue

        jobs.append((row_idx, lemma))

    if limit_triggered and max_jobs is not None:
        logging.info("Reached processing limit of %s entries.", max_jobs)

    workers = max(1, workers)
    rate_limiter = RateLimiter(sleep_seconds) if sleep_seconds > 0 else None

    progress = ProgressTracker(len(jobs), progress_interval) if jobs else None

    user_agent = DEFAULT_USER_AGENT

    logging.info(
        "Scheduling %s lemma(s) across %s worker(s) with minimum %.2fs delay.",
        len(jobs),
        workers,
        sleep_seconds,
    )

    def process_job(row_idx: int, lemma: str) -> Tuple[int, Dict[str, str]]:
        session = get_session(user_agent)
        title = lemma.replace("’", "'").replace(" ", "_")
        wikitext = load_wikitext(
            title,
            session,
            delay=sleep_seconds,
            retries=max_retries,
            cache_dir=cache_dir,
            rate_limiter=rate_limiter,
        )
        registers = detect_registers(wikitext or "")
        return row_idx, registers

    if jobs:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(process_job, row_idx, lemma): row_idx
                for row_idx, lemma in jobs
            }
            for future in as_completed(future_map):
                row_idx, registers = future.result()
                row = rows[row_idx]
                for column, value in registers.items():
                    idx = column_indices[column]
                    if value:
                        row[idx] = value
                    elif overwrite:
                        row[idx] = value
                if progress:
                    progress.advance()

    if progress:
        progress.finish()
        processed = progress.completed
    else:
        processed = 0

    destination = output_path or input_path
    with destination.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)

    if processed:
        logging.info(
            "Finished updating %s entries. Output written to %s.",
            processed,
            destination,
        )
    else:
        logging.info("No rows required updates. Data copied to %s.", destination)

    return processed


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

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

    try:
        update_csv(
            args.input,
            output_path,
            sleep_seconds=args.sleep,
            workers=args.workers,
            progress_interval=args.progress_interval,
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
