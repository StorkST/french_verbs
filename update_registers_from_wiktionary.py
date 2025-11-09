#!/usr/bin/env python3
"""
update_registers_from_wiktionary.py
-----------------------------------

Ajoute les informations de registre (familier / soutenu) aux verbes du fichier CSV
`Verbes_en_français-2025-11-08.csv` en s'appuyant sur les indications présentes
dans le Wiktionnaire français.

Fonctionnement général :
1. Charge la liste des verbes depuis le CSV (colonne `lemme`).
2. Télécharge le wikicode de chaque entrée via l'API Wiktionary.
3. Restreint l'analyse à la section française.
4. Détecte la présence d'étiquettes ou catégories indiquant un registre familier
   ou soutenu (patrons configurables ci-dessous).
5. Met à jour les colonnes `familier` et `soutenu` du CSV (`'X'` si détecté,
   chaîne vide sinon, sauf si une valeur existe déjà).

Le script supporte :
- un cache JSON pour éviter de re-télécharger plusieurs fois la même entrée ;
- une option `--limit` pour tester sur un sous-ensemble ;
- une option `--overwrite` pour écrire directement dans le fichier d'entrée ;
- un mode asynchrone avec limitation du nombre de requêtes simultanées.

Exemple d'utilisation :
    python3 update_registers_from_wiktionary.py \\
        --input Verbes_en_français-2025-11-08.csv \\
        --overwrite

Recommandations :
- Veiller à respecter les règles d'usage du Wiktionnaire (User-Agent identifié,
  volume de requêtes raisonnable).
- Le résultat détecté reste heuristique : un contrôle humain est conseillé
  pour confirmer ou compléter les registres.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import aiohttp
import pandas as pd
import re

# ---------------------------------------------------------------------------
# Configuration générale
# ---------------------------------------------------------------------------

API_URL = "https://fr.wiktionary.org/w/api.php"
USER_AGENT = (
    "JobsAround-VerbRegisters/1.0 "
    "(https://github.com/JobsAround; contact: support@jobsaround.ai)"
)

# Concurrence et temporisation
MAX_CONCURRENT_REQUESTS = 8
REQUEST_TIMEOUT = 40  # secondes
SLEEP_BETWEEN_BATCHES = 0.1  # petite pause pour lisser la charge

# Cache par défaut
DEFAULT_CACHE_PATH = Path("french_verbs/wiktionary_registers_cache.json")

# Patrons pour détecter les registres dans la section française
# (Ils peuvent être adaptés si besoin.)
FAMILIER_PATTERNS = [
    re.compile(r"\{\{\s*(?:registre|étiquette|label|usage|qualifier|lb)\s*\|[^{}]*fam", re.IGNORECASE),
    re.compile(r"\{\{\s*(?:familier|fam)\b[^}]*\}\}", re.IGNORECASE),
    re.compile(r"\[\[\s*catégorie\s*:\s*lexique en français familier", re.IGNORECASE),
    re.compile(r"\bfamili(?:er|èrement)\b", re.IGNORECASE),
]

SOUTENU_PATTERNS = [
    re.compile(r"\{\{\s*(?:registre|étiquette|label|usage|qualifier|lb)\s*\|[^{}]*souten", re.IGNORECASE),
    re.compile(r"\{\{\s*soutenu\b[^}]*\}\}", re.IGNORECASE),
    re.compile(r"\[\[\s*catégorie\s*:\s*lexique en français soutenu", re.IGNORECASE),
    re.compile(r"\bsoutenu\b", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------


def build_title_candidates(lemma: str) -> List[str]:
    """
    Génère plusieurs variantes de titre pour maximiser les chances
    de trouver la page du Wiktionnaire.
    """
    candidates = []
    base = lemma.strip()
    if not base:
        return candidates

    # Variante originale
    candidates.append(base)

    # Variante en minuscules (la plupart des entrées sont en minuscules)
    lower = base.lower()
    if lower not in candidates:
        candidates.append(lower)

    # Variante avec espaces remplacés par des underscores
    underscored = lower.replace(" ", "_")
    if underscored not in candidates:
        candidates.append(underscored)

    # Variante sans apostrophes droites (parfois apostrophes typographiques)
    normalized = underscored.replace("’", "'")
    if normalized not in candidates:
        candidates.append(normalized)

    return candidates


def extract_french_section(wikitext: str) -> str:
    """
    Extrait la section française du wikicode afin de réduire les faux positifs.
    """
    if not wikitext:
        return ""

    # Cherche la section `== {{langue|fr}} ==`
    langue_pattern = re.compile(r"==\s*\{\{langue\|fr\}\}\s*==", re.IGNORECASE)
    match = langue_pattern.search(wikitext)
    if not match:
        return wikitext  # Retourne tout le contenu si la langue n'est pas trouvée

    start = match.end()

    # Cherche la section suivante `== {{langue|...}} ==`
    next_langue = langue_pattern.search(wikitext, pos=start)
    if next_langue:
        return wikitext[start:next_langue.start()]
    return wikitext[start:]


def detect_registers_from_text(text: str) -> Tuple[bool, bool]:
    """
    Retourne un tuple (is_familier, is_soutenu) en analysant la section fournie.
    """
    if not text:
        return False, False

    is_familier = any(pattern.search(text) for pattern in FAMILIER_PATTERNS)
    is_soutenu = any(pattern.search(text) for pattern in SOUTENU_PATTERNS)
    return bool(is_familier), bool(is_soutenu)


async def fetch_wikitext(
    session: aiohttp.ClientSession,
    title: str,
    semaphore: asyncio.Semaphore,
) -> Optional[str]:
    """
    Récupère le wikicode d'un titre donné via l'API du Wiktionnaire.
    """
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
        "titles": title,
    }

    async with semaphore:
        async with session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"[WARNING] HTTP {resp.status} pour '{title}': {text[:120]}...", file=sys.stderr)
                return None
            data = await resp.json()

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None

    page = next(iter(pages.values()))
    revisions = page.get("revisions")
    if not revisions:
        return None

    slots = revisions[0].get("slots", {})
    main = slots.get("main", {})
    return main.get("*")


async def fetch_registers_for_lemma(
    session: aiohttp.ClientSession,
    lemma: str,
    semaphore: asyncio.Semaphore,
) -> Tuple[bool, bool]:
    """
    Tente successivement plusieurs titres jusqu'à trouver le wikicode.
    Retourne (is_familier, is_soutenu).
    """
    for candidate in build_title_candidates(lemma):
        wikitext = await fetch_wikitext(session, candidate, semaphore)
        if wikitext:
            french_section = extract_french_section(wikitext)
            return detect_registers_from_text(french_section)
        await asyncio.sleep(SLEEP_BETWEEN_BATCHES)
    return False, False


def load_cache(path: Path) -> Dict[str, Dict[str, bool]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        print(f"[WARNING] Cache illisible ({path}), il sera recréé.", file=sys.stderr)
    return {}


def save_cache(path: Path, cache: Dict[str, Dict[str, bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)


async def enrich_registers(
    lemmas: List[str],
    cache: Dict[str, Dict[str, bool]],
    limit: Optional[int] = None,
) -> Dict[str, Dict[str, bool]]:
    """
    Rassemble les registres pour chaque lemme (en utilisant le cache si disponible).
    Retourne un mapping {lemme: {'familier': bool, 'soutenu': bool}}.
    """
    results: Dict[str, Dict[str, bool]] = {}
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = []
        count = 0

        for lemma in lemmas:
            if limit is not None and count >= limit:
                break
            count += 1

            if lemma in cache:
                results[lemma] = cache[lemma]
                continue

            tasks.append((lemma, asyncio.create_task(fetch_registers_for_lemma(session, lemma, semaphore))))

        for lemma, task in tasks:
            try:
                is_familier, is_soutenu = await task
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Échec pour '{lemma}': {exc}", file=sys.stderr)
                is_familier = is_soutenu = False

            record = {"familier": bool(is_familier), "soutenu": bool(is_soutenu)}
            results[lemma] = record
            cache[lemma] = record  # mise à jour du cache

    return results


def apply_registers_to_dataframe(
    df: pd.DataFrame,
    registers: Dict[str, Dict[str, bool]],
    *,
    overwrite_existing: bool = False,
) -> pd.DataFrame:
    """
    Met à jour les colonnes `familier` / `soutenu` selon les informations fournies.
    """
    for idx, row in df.iterrows():
        lemma = str(row["lemme"])
        if lemma not in registers:
            continue

        info = registers[lemma]
        if info.get("familier"):
            if overwrite_existing or not str(row.get("familier", "")).strip():
                df.at[idx, "familier"] = "X"
        elif overwrite_existing:
            df.at[idx, "familier"] = ""

        if info.get("soutenu"):
            if overwrite_existing or not str(row.get("soutenu", "")).strip():
                df.at[idx, "soutenu"] = "X"
        elif overwrite_existing:
            df.at[idx, "soutenu"] = ""

    return df


def _is_marked(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complète les colonnes 'familier' et 'soutenu' à partir du Wiktionnaire."
    )
    parser.add_argument(
        "--input",
        default="french_verbs/Verbes_en_français-2025-11-08.csv",
        help="Chemin du CSV d'entrée (par défaut: %(default)s).",
    )
    parser.add_argument(
        "--output",
        help="Chemin du CSV de sortie. Si non spécifié, crée un fichier *_with_registers.csv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Écrase le fichier d'entrée au lieu d'écrire une copie.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Nombre maximum de verbes à traiter (pour tests).",
    )
    parser.add_argument(
        "--cache",
        default=str(DEFAULT_CACHE_PATH),
        help="Chemin du fichier cache JSON (par défaut: %(default)s).",
    )
    parser.add_argument(
        "--skip-populated",
        action="store_true",
        help="Ignore les verbes dont les colonnes 'familier' et 'soutenu' sont déjà renseignées.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore le cache existant et récupère à nouveau toutes les entrées.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERREUR] Fichier introuvable: {input_path}", file=sys.stderr)
        sys.exit(1)

    cache_path = Path(args.cache)
    cache: Dict[str, Dict[str, bool]] = {} if args.force_refresh else load_cache(cache_path)

    print(f"[INFO] Chargement du CSV: {input_path}")
    df = pd.read_csv(input_path)

    # Vérifie la présence des colonnes (créées en amont)
    for col in ("familier", "soutenu"):
        if col not in df.columns:
            print(f"[INFO] Colonne '{col}' absente, initialisation à vide.")
            df[col] = ""

    df["familier"] = df["familier"].fillna("").astype(str)
    df["soutenu"] = df["soutenu"].fillna("").astype(str)

    lemmas: List[str] = []
    familier_col = df.get("familier")
    soutenu_col = df.get("soutenu")

    for idx, lemma in enumerate(df["lemme"]):
        populated = _is_marked(familier_col.iloc[idx]) or _is_marked(soutenu_col.iloc[idx])
        if args.skip_populated and populated:
            continue
        if isinstance(lemma, str) and lemma.strip():
            lemmas.append(lemma.strip())

    if args.limit is not None:
        lemmas = lemmas[: args.limit]

    print(f"[INFO] Nombre de verbes à traiter: {len(lemmas)} (limite={args.limit})")

    start_time = time.time()
    updated_cache = cache.copy()
    registers: Dict[str, Dict[str, bool]] = {}

    if lemmas:
        registers = asyncio.run(enrich_registers(lemmas, updated_cache, limit=None))
        save_cache(cache_path, updated_cache)

    df = apply_registers_to_dataframe(df, registers, overwrite_existing=args.overwrite)

    if args.overwrite:
        output_path = input_path
    else:
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.with_name(f"{input_path.stem}_with_registers{input_path.suffix}")

    df.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    print(f"[INFO] Terminé en {elapsed:.1f}s. Résultats écrits dans: {output_path}")
    print(
        f"[INFO] Registres détectés: "
        f"familier={sum(df['familier'].astype(str).str.upper().eq('X'))} | "
        f"soutenu={sum(df['soutenu'].astype(str).str.upper().eq('X'))}"
    )


if __name__ == "__main__":
    main()

