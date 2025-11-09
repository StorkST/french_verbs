#!/usr/bin/env python3
"""
Batch tag French verbs as informal, formal or neutral using Grok.

The script reads a CSV containing a `lemme` column, sends the verbs to Grok in
small batches, and updates/creates the columns `Category_Informal` and
`Category_Formal` based on the model's response. Each verb is classified as
either `informal`, `formal`, or `neutral`. Only one register can be set to "X";
neutral means both columns stay blank.

Environment:
    GROK_URL      (default: https://api.x.ai/v1/chat/completions)
    GROK_API_KEY  (required)
    GROK_MODEL    (default: grok-4)

Example:
    python tag_registers_with_grok.py \\
        --input Verbes_en_français-2025-11-09.csv \\
        --output Verbes_en_français-2025-11-09.csv \\
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
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import aiohttp


GROK_URL = os.getenv("GROK_URL", "https://api.x.ai/v1/chat/completions")
GROK_API_KEY = os.getenv("GROK_API_KEY", "API-key")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4")

DEFAULT_BATCH_SIZE = 30
DEFAULT_MAX_CONCURRENT = 5
DEFAULT_RPM = 250

SYSTEM_PROMPT = (
    "Tu es un lexicographe spécialisé dans les registres du français. "
    "Pour chaque verbe fourni, indique s'il est généralement employé dans un "
    "registre informel (familier, populaire, argotique), dans un registre "
    "formel/soutenu, ou si son usage courant est neutre. "
    "Ne classe un verbe comme informel ou formel que si le verbe est "
    "reconnu comme tel dans les grands dictionnaires contemporains "
    "(Le Robert, Larousse, Bescherelle, Antidote, TLFi, etc.). "
    "Si le verbe possède plusieurs sens mais n'est informel ou formel que "
    "dans certains emplois très limités, choisis la dominante la plus neutre. "
    "Si tu n'es pas certain, choisis \"neutral\"."
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
    return (
        "Classe les verbes suivants :\n"
        f"{items}\n\n"
        "Signale pour chaque verbe uniquement l'une de ces étiquettes :\n"
        '  - "informal" (familier / argot / populaire)\n'
        '  - "formal" (soutenu / littéraire)\n'
        '  - "neutral"\n'
        "Réponds avec un JSON strict du type :\n"
        '[{"verb": "verbe", "register": "informal"}]\n'
        "N'ajoute aucun texte autour. Conserve exactement l'orthographe du verbe fourni."
    )


async def classify_batch(
    session: aiohttp.ClientSession,
    verbs: Sequence[str],
    semaphore: asyncio.Semaphore,
    limiter: RateLimiter,
    model: str,
    max_retries: int = 3,
    batch_id: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, str]:
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
                raise ValueError("Response JSON is not a list")
            result: Dict[str, str] = {}
            if verbose:
                print(
                    f"[DEBUG] Batch {batch_id or '?'} returned JSON with "
                    f"{len(parsed)} entries."
                )
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                verb = item.get("verb")
                register = item.get("register")
                if (
                    isinstance(verb, str)
                    and isinstance(register, str)
                    and register in {"informal", "formal", "neutral"}
                ):
                    result[verb.strip()] = register
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
) -> Dict[str, str]:
    limiter = RateLimiter(rpm, verbose=verbose)
    semaphore = asyncio.Semaphore(max_concurrent)

    results: Dict[str, str] = {}
    total_batches = math.ceil(len(verbs) / batch_size)
    total_verbs = len(verbs)
    processed_verbs = 0
    informal_total = 0
    formal_total = 0

    async with aiohttp.ClientSession() as session:
        # Prepare wrapper tasks that keep batch metadata with their results
        tasks: List[asyncio.Task[Tuple[int, List[str], Dict[str, str]]]] = []

        async def run_batch(batch_index: int, batch_verbs: List[str]) -> Tuple[int, List[str], Dict[str, str]]:
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
                    informal_total += sum(
                        1 for register in batch_result.values() if register == "informal"
                    )
                    formal_total += sum(
                        1 for register in batch_result.values() if register == "formal"
                    )
                    percent = processed_verbs / total_verbs * 100 if total_verbs else 100
                    print(
                        f"[{index}/{total_batches}] "
                        f"Processed {processed_verbs}/{total_verbs} verbs "
                        f"({percent:.1f}%). "
                        f"Informal so far: {informal_total}, formal so far: {formal_total}."
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
    informal_total = 0
    formal_total = 0

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
                ordered = [{"verb": verb, "register": result[verb]} for verb in batch]
                print(json.dumps(ordered, ensure_ascii=False, indent=2))
                informal_total += sum(
                    1 for register in result.values() if register == "informal"
                )
                formal_total += sum(
                    1 for register in result.values() if register == "formal"
                )
            else:
                print("[INFO] Aucun résultat reçu.")

            processed_verbs += len(batch)
            percent = processed_verbs / total_verbs * 100 if total_verbs else 100
            print(
                f"Progression : {processed_verbs}/{total_verbs} verbes "
                f"({percent:.1f}%). Informel cumulés : {informal_total}, "
                f"soutenu cumulés : {formal_total}."
            )

            user_input = await asyncio.to_thread(
                input,
                "Appuyez sur Entrée pour continuer (q pour quitter) : ",
            )
            if user_input.strip().lower().startswith("q"):
                print("Arrêt demandé par l'utilisateur.")
                break


def update_csv(
    input_path: Path,
    output_path: Path,
    classifications: Dict[str, str],
) -> Tuple[int, int, int, int, int]:
    with input_path.open("r", newline="", encoding="utf-8") as fh:
        reader = list(csv.reader(fh))
    if not reader:
        raise SystemExit(f"CSV is empty: {input_path}")

    header = reader[0]
    try:
        lemma_idx = header.index("lemme")
    except ValueError as exc:
        raise SystemExit("CSV must contain a 'lemme' column") from exc

    def ensure_column(name: str) -> int:
        if name not in header:
            header.append(name)
        return header.index(name)

    informal_idx = ensure_column("Category_Informal")
    formal_idx = ensure_column("Category_Formal")

    updated = 0
    informal_total = 0
    formal_total = 0
    neutral_total = 0
    unknown = 0

    for row in reader[1:]:
        if len(row) <= max(informal_idx, formal_idx, lemma_idx):
            row.extend([""] * (max(informal_idx, formal_idx, lemma_idx) + 1 - len(row)))
        lemma = row[lemma_idx].strip()
        if not lemma:
            continue
        register = classifications.get(lemma)
        if not register:
            unknown += 1
            continue
        row[informal_idx] = "X" if register == "informal" else ""
        row[formal_idx] = "X" if register == "formal" else ""
        updated += 1
        if register == "informal":
            informal_total += 1
        elif register == "formal":
            formal_total += 1
        else:
            neutral_total += 1

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(reader)

    return updated, informal_total, formal_total, neutral_total, unknown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use Grok to tag French verbs by register."
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
        help="Destination CSV (default: overwrite input).",
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
        help="Skip verbs that already have a register assigned.",
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


def main() -> int:
    args = parse_args()
    ensure_api_key()

    input_path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    if args.output:
        output_path = args.output
    else:
        suffix = "_formal-registries"
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
            if args.skip_existing and (
                (row.get("Category_Informal") or "").strip()
                or (row.get("Category_Formal") or "").strip()
            ):
                continue
            verbs.append(lemma)

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

    (
        updated,
        informal_total,
        formal_total,
        neutral_total,
        missing,
    ) = update_csv(input_path, output_path, classifications)

    print(f"Updated {updated} verbs.")
    print(
        f"Breakdown → informal: {informal_total}, formal: {formal_total}, "
        f"neutral: {neutral_total}"
    )
    if missing:
        print(f"Warning: {missing} verbs had no classification from Grok.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

