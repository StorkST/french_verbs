#!/usr/bin/env python3
"""
Batch tag French verbs with employment-related categories using Grok.

The script reads a CSV containing a `lemme` column, sends the verbs to Grok in
small batches, and updates/creates the category columns listed in
`CATEGORY_DEFINITIONS`. Each verb can belong to zero, one, or multiple
categories. The model must answer with a JSON list where each entry is a dict
containing a verb and the corresponding list of category identifiers.

Environment:
    GROK_URL      (default: https://api.x.ai/v1/chat/completions)
    GROK_API_KEY  (required)
    GROK_MODEL    (default: grok-4)

Example:
    python tag_job_categories_with_grok.py \\
        --input Verbes_en_français-2025-11-09.csv \\
        --output Verbes_en_français-2025-11-09_categories.csv \\
        --batch-size 30 --limit 300
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import aiohttp


GROK_URL = os.getenv("GROK_URL", "https://api.x.ai/v1/chat/completions")
GROK_API_KEY = os.getenv("GROK_API_KEY", "API-key")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4")

DEFAULT_BATCH_SIZE = 30
DEFAULT_MAX_CONCURRENT = 8
DEFAULT_RPM = 300
ROW_PROGRESS_INTERVAL = 200

CATEGORY_DEFINITIONS: Sequence[Tuple[str, str]] = [
    (
        "Category_Administrative_Legal",
        "Travail administratif, juridique, gestion de dossiers, conformité.",
    ),
    (
        "Category_Office_Work",
        "Travail de bureau général, gestion, coordination, tâches cléricales.",
    ),
    (
        "Category_Customer_Service",
        "Relation clientèle, assistance, vente au comptoir ou au téléphone.",
    ),
    (
        "Category_Manufacturing_Industry",
        "Production industrielle, fabrication, exploitation d'équipements.",
    ),
    (
        "Category_Construction_Work",
        "Chantiers, construction, travaux publics, métiers du bâtiment.",
    ),
    (
        "Category_Maintenance_Repair",
        "Entretien, réparation, dépannage, maintenance technique.",
    ),
    (
        "Category_Cooking_Catering",
        "Cuisine, restauration, préparation alimentaire, traiteur.",
    ),
    (
        "Category_Health_Medical",
        "Soins de santé, médical, paramédical, accompagnement patients.",
    ),
    (
        "Category_Tourism_Hospitality",
        "Tourisme, hôtellerie, accueil, loisirs, animation.",
    ),
    (
        "Category_Sports",
        "Sports, entraînement, coaching, activités physiques.",
    ),
]

CATEGORY_NAMES = [name for name, _ in CATEGORY_DEFINITIONS]
CATEGORY_SET = set(CATEGORY_NAMES)

SYSTEM_PROMPT = (
    "Tu es un lexicographe spécialisé dans les usages professionnels des verbes. "
    "Pour chaque verbe fourni, identifie les contextes métiers auxquels il est "
    "couramment associé. Utilise uniquement les catégories fournies. "
    "Si aucune catégorie ne convient clairement, renvoie une liste vide. "
    "Ne devine pas à partir de simples associations faibles : concentre-toi sur "
    "les usages métiers courants et explicites."
)


def ensure_api_key() -> None:
    if not GROK_API_KEY:
        raise SystemExit(
            "GROK_API_KEY environment variable must be set. "
            "Export your key first."
        )


class RateLimiter:
    """Simple asynchronous rate limiter (requests per minute)."""

    def __init__(
        self,
        max_requests: int,
        interval: float = 60.0,
        *,
        verbose: bool = False,
    ) -> None:
        self.max_requests = max_requests
        self.interval = interval
        self.timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()
        self.verbose = verbose

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            while self.timestamps and self.timestamps[0] < now - self.interval:
                self.timestamps.popleft()
            if len(self.timestamps) >= self.max_requests:
                sleep_for = self.interval - (now - self.timestamps[0]) + 0.01
                if self.verbose:
                    print(
                        f"[RateLimiter] Waiting {sleep_for:.2f}s to respect RPM limit."
                    )
                await asyncio.sleep(max(sleep_for, 0))
                now = time.monotonic()
                while self.timestamps and self.timestamps[0] < now - self.interval:
                    self.timestamps.popleft()
            self.timestamps.append(time.monotonic())


def chunked(iterable: Sequence[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(iterable), size):
        yield list(iterable[idx : idx + size])


def build_user_prompt(batch: Sequence[str]) -> str:
    items = "\n".join(f"- {verb}" for verb in batch)
    category_lines = "\n".join(
        f'  - "{name}": {description}'
        for name, description in CATEGORY_DEFINITIONS
    )
    return (
        "Analyse les verbes suivants :\n"
        f"{items}\n\n"
        "Pour chaque verbe, retourne les catégories métiers correspondantes parmi :\n"
        f"{category_lines}\n\n"
        "Réponds EXCLUSIVEMENT avec un JSON strict de ce type :\n"
        '[{"verb": "verbe", "categories": ["Category_Office_Work", "Category_Customer_Service"]}]\n'
        "La clé \"categories\" doit être une liste (éventuellement vide) contenant uniquement "
        "les identifiants de catégories indiqués. N'ajoute aucun texte autour et conserve "
        "exactement l'orthographe du verbe fourni."
    )


def normalize_categories(raw_categories: object) -> List[str]:
    if raw_categories is None:
        return []
    if isinstance(raw_categories, str):
        candidate = raw_categories.strip()
        if not candidate or candidate.lower() in {"", "none", "neutral"}:
            return []
        raw_list = [candidate]
    elif isinstance(raw_categories, (list, tuple)):
        raw_list = list(raw_categories)
    else:
        raise ValueError("Categories must be a list of strings or an empty value.")

    normalized: List[str] = []
    for entry in raw_list:
        if not isinstance(entry, str):
            continue
        candidate = entry.strip()
        if candidate in CATEGORY_SET and candidate not in normalized:
            normalized.append(candidate)
    return normalized


async def classify_batch(
    session: aiohttp.ClientSession,
    verbs: Sequence[str],
    semaphore: asyncio.Semaphore,
    limiter: RateLimiter,
    model: str,
    max_retries: int = 3,
    batch_id: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, List[str]]:
    attempt = 0
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(verbs)},
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
    }

    while attempt <= max_retries:
        async with semaphore:
            await limiter.acquire()
            try:
                preview = ", ".join(verbs[:5])
                if len(verbs) > 5:
                    preview += ", ..."
                print(
                    f"[Batch {batch_id or '?'}] Attempt {attempt + 1}: "
                    f"Sending {len(verbs)} verbs ({preview})",
                    flush=True,
                )
                if verbose:
                    print(
                        f"[DEBUG] Batch {batch_id or '?'} starting attempt "
                        f"{attempt + 1} (verbs: {', '.join(verbs[:5])}"
                        f"{'...' if len(verbs) > 5 else ''})"
                    )
                async with session.post(
                    GROK_URL,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {GROK_API_KEY}",
                    },
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    if response.status >= 500 or response.status in {429}:
                        attempt += 1
                        await asyncio.sleep(2 * attempt)
                        continue
                    if response.status != 200:
                        text = await response.text()
                        raise RuntimeError(
                            f"Grok HTTP {response.status}: {text[:200]}"
                        )
                    data = await response.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(f"Request failed after retries: {exc}") from exc
                if verbose:
                    print(
                        f"[DEBUG] Batch {batch_id or '?'} encountered "
                        f"{type(exc).__name__}; retrying in {2 * attempt}s."
                    )
                await asyncio.sleep(2 * attempt)
                continue

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(lines[1:-1]).strip()
        try:
            parsed = json.loads(content)
            if not isinstance(parsed, list):
                raise ValueError("Response JSON is not a list.")
            result: Dict[str, List[str]] = {}
            if verbose:
                print(
                    f"[DEBUG] Batch {batch_id or '?'} returned JSON with "
                    f"{len(parsed)} entries."
                )
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                verb = item.get("verb")
                categories = normalize_categories(item.get("categories"))
                if isinstance(verb, str) and verb.strip():
                    result[verb.strip()] = categories
            missing = [v for v in verbs if v not in result]
            if missing:
                raise ValueError(
                    f"Missing verbs in response: {missing}; raw={content[:200]}"
                )
            return result
        except (json.JSONDecodeError, ValueError) as exc:
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(f"Failed to parse response: {exc}") from exc
            if verbose:
                print(
                    f"[DEBUG] Batch {batch_id or '?'} parse error: {exc}; "
                    f"retrying in {2 * attempt}s."
                )
            await asyncio.sleep(2 * attempt)

    raise RuntimeError("Unreachable")


class BatchClassificationError(RuntimeError):
    def __init__(self, index: int, batch: Sequence[str], reason: Exception) -> None:
        super().__init__(str(reason))
        self.index = index
        self.batch = list(batch)
        self.reason = reason


async def classify_verbs(
    verbs: Sequence[str],
    batch_size: int,
    max_concurrent: int,
    rpm: int,
    model: str,
    *,
    verbose: bool = False,
) -> Dict[str, List[str]]:
    limiter = RateLimiter(rpm, verbose=verbose)
    semaphore = asyncio.Semaphore(max_concurrent)

    results: Dict[str, List[str]] = {}
    total_batches = math.ceil(len(verbs) / batch_size)
    total_verbs = len(verbs)
    processed_verbs = 0
    category_totals: Counter[str] = Counter()

    async with aiohttp.ClientSession() as session:
        tasks: List[asyncio.Task[Tuple[int, List[str], Dict[str, List[str]]]]] = []

        async def run_batch(
            batch_index: int, batch_verbs: List[str]
        ) -> Tuple[int, List[str], Dict[str, List[str]]]:
            try:
                result = await classify_batch(
                    session,
                    batch_verbs,
                    semaphore,
                    limiter,
                    model,
                    batch_id=batch_index,
                    verbose=verbose,
                )
                return batch_index, batch_verbs, result
            except Exception as exc:  # noqa: BLE001
                raise BatchClassificationError(batch_index, batch_verbs, exc) from exc

        for index, batch in enumerate(chunked(list(verbs), batch_size), start=1):
            task = asyncio.create_task(run_batch(index, list(batch)))
            tasks.append(task)

        try:
            for finished in asyncio.as_completed(tasks):
                try:
                    index, batch, batch_result = await finished
                    results.update(batch_result)
                    processed_verbs += len(batch)
                    batch_counts: Counter[str] = Counter()
                    for categories in batch_result.values():
                        batch_counts.update(categories)
                    category_totals.update(batch_counts)
                    percent = processed_verbs / total_verbs * 100 if total_verbs else 100
                    preview_items = []
                    for verb in list(batch)[:5]:
                        categories = batch_result.get(verb, [])
                        if categories:
                            preview_items.append(f"{verb}:{'|'.join(categories)}")
                        else:
                            preview_items.append(f"{verb}:-")
                    preview = ", ".join(preview_items)
                    batch_summary = ", ".join(
                        f"{name}:{batch_counts.get(name, 0)}"
                        for name in CATEGORY_NAMES
                        if batch_counts.get(name, 0)
                    ) or "none"
                    cumulative_summary = ", ".join(
                        f"{name}:{category_totals.get(name, 0)}"
                        for name in CATEGORY_NAMES
                        if category_totals.get(name, 0)
                    ) or "none"
                    print(
                        f"[Batch {index}/{total_batches}] "
                        f"Progress {processed_verbs}/{total_verbs} verbs "
                        f"({percent:.1f}%). "
                        f"Batch counts → {batch_summary}. "
                        f"Cumulative totals → {cumulative_summary}. "
                        f"Sample → {preview}.",
                        flush=True,
                    )
                except BatchClassificationError as exc:
                    index = exc.index
                    batch = exc.batch
                    print(
                        f"[ERROR] Batch {index} failed ({', '.join(batch)}): {exc.reason}",
                        file=sys.stderr,
                    )
                    processed_verbs += len(batch)
                    percent = processed_verbs / total_verbs * 100 if total_verbs else 100
                    print(
                        f"[WARNING] Marking batch {index} as missing. "
                        f"Progress {processed_verbs}/{total_verbs} verbs "
                        f"({percent:.1f}%)."
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[ERROR] Unexpected failure: {exc}", file=sys.stderr)
        finally:
            pending_tasks = [task for task in tasks if not task.done()]
            for pending in pending_tasks:
                pending.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
    return results


async def classify_verbs_one_step(
    verbs: Sequence[str],
    batch_size: int,
    rpm: int,
    model: str,
    *,
    verbose: bool = False,
) -> None:
    limiter = RateLimiter(rpm, verbose=verbose)
    semaphore = asyncio.Semaphore(1)
    total_batches = math.ceil(len(verbs) / batch_size)
    total_verbs = len(verbs)
    processed_verbs = 0
    category_totals: Counter[str] = Counter()

    async with aiohttp.ClientSession() as session:
        for index, batch in enumerate(chunked(list(verbs), batch_size), start=1):
            print(f"\n[Batch {index}/{total_batches}] {', '.join(batch)}")
            try:
                result = await classify_batch(
                    session,
                    batch,
                    semaphore,
                    limiter,
                    model,
                    batch_id=index,
                    verbose=verbose,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Échec de la classification : {exc}")
                result = {}

            if result:
                ordered = [{"verb": verb, "categories": result[verb]} for verb in batch]
                print(json.dumps(ordered, ensure_ascii=False, indent=2))
                batch_counts: Counter[str] = Counter()
                for categories in result.values():
                    batch_counts.update(categories)
                category_totals.update(batch_counts)
                batch_summary = ", ".join(
                    f"{name}:{batch_counts.get(name, 0)}"
                    for name in CATEGORY_NAMES
                    if batch_counts.get(name, 0)
                ) or "none"
            else:
                print("[INFO] Aucun résultat reçu.")
                batch_summary = "none"

            processed_verbs += len(batch)
            percent = processed_verbs / total_verbs * 100 if total_verbs else 100
            summary = ", ".join(
                f"{name}:{category_totals.get(name, 0)}"
                for name in CATEGORY_NAMES
                if category_totals.get(name, 0)
            ) or "none"
            print(
                f"Progression : {processed_verbs}/{total_verbs} verbes "
                f"({percent:.1f}%). Batch → {batch_summary}. Totaux → {summary}."
            )

            user_input = await asyncio.to_thread(
                input,
                "Appuyez sur Entrée pour continuer (q pour quitter) : ",
            )
            if user_input.strip().lower().startswith("q"):
                print("Arrêt demandé par l'utilisateur.")
                break


def ensure_columns(header: List[str]) -> Dict[str, int]:
    indices: Dict[str, int] = {}
    for name in CATEGORY_NAMES:
        if name not in header:
            header.append(name)
        indices[name] = header.index(name)
    return indices


def update_csv(
    input_path: Path,
    output_path: Path,
    classifications: Dict[str, List[str]],
) -> Tuple[int, Dict[str, int], int]:
    with input_path.open("r", newline="", encoding="utf-8") as fh:
        reader = list(csv.reader(fh))
    if not reader:
        raise SystemExit(f"CSV is empty: {input_path}")

    header = reader[0]
    try:
        lemma_idx = header.index("lemme")
    except ValueError as exc:
        raise SystemExit("CSV must contain a 'lemme' column") from exc

    column_indices = ensure_columns(header)

    updated_rows = 0
    category_totals = {name: 0 for name in CATEGORY_NAMES}
    missing = 0

    total_rows = max(len(reader) - 1, 0)
    processed_rows = 0

    for row in reader[1:]:
        target_len = max(column_indices.values(), default=0)
        if len(row) <= max(target_len, lemma_idx):
            row.extend([""] * (max(target_len, lemma_idx) + 1 - len(row)))
        lemma = row[lemma_idx].strip()
        if not lemma:
            processed_rows += 1
            if (
                total_rows
                and processed_rows % ROW_PROGRESS_INTERVAL == 0
            ):
                print(
                    f"[update_csv] Processed {processed_rows}/{total_rows} rows.",
                    flush=True,
                )
            continue
        categories = classifications.get(lemma)
        if categories is None:
            missing += 1
            processed_rows += 1
            if (
                total_rows
                and processed_rows % ROW_PROGRESS_INTERVAL == 0
            ):
                print(
                    f"[update_csv] Processed {processed_rows}/{total_rows} rows.",
                    flush=True,
                )
            continue
        predicted = set(categories)
        row_updated = False

        for name in CATEGORY_NAMES:
            idx = column_indices[name]
            existing_value = row[idx].strip()
            if name in predicted:
                if existing_value != "X":
                    row[idx] = "X"
                    row_updated = True
                category_totals[name] += 1
            else:
                if name == "Category_Manufacturing_Industry":
                    # Preserve existing assignments for manufacturing.
                    continue
                if existing_value:
                    row[idx] = ""
                    row_updated = True

        if row_updated:
            updated_rows += 1

        processed_rows += 1
        if total_rows and processed_rows % ROW_PROGRESS_INTERVAL == 0:
            print(
                f"[update_csv] Processed {processed_rows}/{total_rows} rows.",
                flush=True,
            )

    if total_rows and processed_rows % ROW_PROGRESS_INTERVAL != 0:
        print(
            f"[update_csv] Processed {processed_rows}/{total_rows} rows.",
            flush=True,
        )

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(reader)

    return updated_rows, category_totals, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use Grok to tag French verbs with employment-related categories."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Verbes_en_français-2025-11-09.csv"),
        help="Source CSV with a 'lemme' column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination CSV (default: input name with '_job-categories').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of verbs per Grok request.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help="Maximum concurrent Grok requests.",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=DEFAULT_RPM,
        help="Request rate limit (per minute).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N verbs (for testing).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip verbs that already have at least one category assigned.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed debug information about API calls and rate limiting.",
    )
    parser.add_argument(
        "--one-step",
        action="store_true",
        help="Interactive mode: show each batch result in the console and wait for Enter.",
    )
    return parser.parse_args()


def has_existing_category(row: Dict[str, str]) -> bool:
    return any((row.get(name) or "").strip() for name in CATEGORY_NAMES)


def main() -> int:
    args = parse_args()
    ensure_api_key()

    input_path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    print(f"[main] Reading verbs from {input_path} ...")

    if args.output:
        output_path = args.output
    else:
        suffix = "_job-categories"
        if input_path.suffix:
            output_path = input_path.with_name(
                f"{input_path.stem}{suffix}{input_path.suffix}"
            )
        else:
            output_path = input_path.with_name(f"{input_path.name}{suffix}")

    with input_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        verbs = []
        for row in reader:
            lemma = (row.get("lemme") or "").strip()
            if not lemma:
                continue
            if args.skip_existing and has_existing_category(row):
                continue
            verbs.append(lemma)

    print(
        f"[main] Prepared {len(verbs)} verbs for classification "
        f"(skip_existing={args.skip_existing})."
    )

    if args.limit is not None:
        verbs = verbs[: args.limit]

    if not verbs:
        print("No verbs to classify.")
        return 0

    if args.one_step:
        asyncio.run(
            classify_verbs_one_step(
                verbs,
                batch_size=args.batch_size,
                rpm=args.rpm,
                model=GROK_MODEL,
                verbose=args.verbose,
            )
        )
        return 0

    print(
        f"Classifying {len(verbs)} verbs using Grok ({args.batch_size} per batch, "
        f"{args.max_concurrent} concurrent, {args.rpm} rpm)."
    )

    classifications = asyncio.run(
        classify_verbs(
            verbs,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent,
            rpm=args.rpm,
            model=GROK_MODEL,
            verbose=args.verbose,
        )
    )

    print(f"[main] Applying classifications to {output_path} ...")

    updated_rows, category_totals, missing = update_csv(
        input_path,
        output_path,
        classifications,
    )

    print(f"Updated {updated_rows} rows.")
    breakdown = ", ".join(
        f"{name}: {count}" for name, count in category_totals.items()
    )
    print(f"Breakdown → {breakdown}")
    if missing:
        print(f"Warning: {missing} verbs had no classification from Grok.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


