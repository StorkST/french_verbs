#!/usr/bin/env python3
"""Build a register lexicon from Wiktionary category listings.

This helper queries the French Wiktionary API for category members such as
``Catégorie:Verbes en français familier`` or ``Catégorie:Verbes en français
soutenu`` and produces a JSON lexicon compatible with
``assign_registers_from_lexicon.py``.

Usage examples
--------------

Fetch verbs present in the main CSV and overwrite the default lexicon::

    python build_register_lexicon_from_categories.py \
        --filter-csv Verbes_en_français-2025-11-09.csv \
        --output register_lexicon.json

Generate a new lexicon file without filtering::

    python build_register_lexicon_from_categories.py --output lexicon.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import requests

WIKTIONARY_API = "https://fr.wiktionary.org/w/api.php"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "register_lexicon.json"
DEFAULT_USER_AGENT = (
    "RegisterLexiconBuilder/1.0 (https://github.com/JobsAround; contact: data-team@jobsaround.example)"
)

CATEGORY_MAP: Dict[str, Dict[str, str]] = {
    "informal": {
        "Catégorie:Termes familiers en français": "familier",
        "Catégorie:Termes très familiers en français": "très familier",
        "Catégorie:Termes populaires en français": "populaire",
        "Catégorie:Termes argotiques en français": "argotique",
    },
    "formal": {
        "Catégorie:Termes soutenus en français": "soutenu",
        "Catégorie:Termes littéraires en français": "littéraire",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an informal/formal lexicon from Wiktionary categories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSON lexicon file.",
    )
    parser.add_argument(
        "--filter-csv",
        type=Path,
        help="Limit verbs to those present in this CSV (expects a 'lemme' column).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge results with an existing lexicon file instead of overwriting.",
    )
    return parser.parse_args()


def load_lemmas_from_csv(path: Path) -> Set[str]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        lemmas = {row["lemme"].strip() for row in reader if row.get("lemme")}
    return lemmas


def fetch_category_members(title: str, session: requests.Session) -> Set[str]:
    members: Set[str] = set()
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": title,
        "cmlimit": 500,
        "cmtype": "page",
        "format": "json",
    }

    while True:
        logging.debug("Requesting members of %s", title)
        response = session.get(WIKTIONARY_API, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        for item in data.get("query", {}).get("categorymembers", []):
            page_title = item.get("title")
            if not page_title:
                continue
            page_title = page_title.strip()
            if "#" in page_title:
                continue
            members.add(page_title.replace(" ", " "))
        cont = data.get("continue")
        if not cont:
            break
        params.update(cont)
    return members


def normalise(terms: Iterable[str]) -> Set[str]:
    result = set()
    for term in terms:
        base = term.split(" (")[0]
        base = base.strip()
        if not base:
            continue
        result.add(base)
    return result


def merge_existing(path: Path, lexicon: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    if not path.exists():
        return lexicon
    data = json.loads(path.read_text(encoding="utf-8"))
    merged = {
        "informal": set(data.get("informal", [])),
        "formal": set(data.get("formal", [])),
    }
    merged["informal"].update(lexicon["informal"])
    merged["formal"].update(lexicon["formal"])
    return merged


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    filter_lemmas: Optional[Set[str]] = None
    if args.filter_csv:
        logging.info("Filtering results using lemmas from %s", args.filter_csv)
        filter_lemmas = load_lemmas_from_csv(args.filter_csv)

    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})

    collected: Dict[str, Set[str]] = {"informal": set(), "formal": set()}

    for register, categories in CATEGORY_MAP.items():
        for category_title, label in categories.items():
            logging.info("Fetching %s verbs from %s", register, category_title)
            members = fetch_category_members(category_title, session)
            normalised = normalise(members)
            if filter_lemmas is not None:
                normalised &= filter_lemmas
            collected[register].update(normalised)
            logging.info(
                "  %s entries (after filtering) labelled %s", len(normalised), label
            )

    for register in collected:
        collected[register] = set(sorted(collected[register]))

    if args.merge:
        collected = merge_existing(args.output, collected)

    overlap = collected["informal"] & collected["formal"]
    if overlap:
        logging.warning(
            "Removing %d verbs that appear in both registers: %s",
            len(overlap),
            ", ".join(sorted(list(overlap))[:10]),
        )
        collected["informal"] -= overlap
        collected["formal"] -= overlap

    args.output.write_text(
        json.dumps(
            {
                "informal": sorted(collected["informal"]),
                "formal": sorted(collected["formal"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logging.info(
        "Wrote %d informal and %d formal verbs to %s",
        len(collected["informal"]),
        len(collected["formal"]),
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
