"""Batch process many article URLs into sentence records with player attribution."""

import argparse
from collections import defaultdict
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import spacy

from article_player_split import (
    create_player_matcher,
    fetch_text,
    load_alias_map,
    normalize,
)
from pronoun_backfill import build_pronoun_backfill_record, sentence_has_pronoun


def load_done_urls_from_jsonl(path: Path) -> Set[str]:
    """Load URLs already processed from a JSONL output (supports optional url_done markers)."""
    done_urls: Set[str] = set()
    seen_urls: Set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                url = obj.get("url")
                if url:
                    seen_urls.add(str(url))
                if obj.get("_type") == "url_done" and url:
                    done_urls.add(str(url))
    except FileNotFoundError:
        return set()
    except OSError:
        return set()

    return done_urls if done_urls else seen_urls


def assert_safe_output_path(format_name: str, output_path: Path, force: bool) -> None:
    """Prevent accidentally overwriting grouped JSON with JSONL (and vice versa)."""
    if force or not output_path.exists():
        return

    suffix = output_path.suffix.lower()
    name = output_path.name.lower()

    if format_name == "jsonl":
        if suffix == ".json" or name.endswith("_by_player.json"):
            raise ValueError(
                f"Refusing to overwrite existing JSON file with JSONL output: {output_path}. "
                "Choose a .jsonl output path or pass --force."
            )
        return

    if format_name == "grouped-json":
        if suffix == ".jsonl":
            raise ValueError(
                f"Refusing to overwrite existing JSONL file with grouped JSON output: {output_path}. "
                "Choose a .json output path or pass --force."
            )


def pick_primary_player(spans: Sequence[Tuple[str, int, int]]) -> Tuple[Optional[str], str]:
    """Tie-break player mentions to pick one primary target."""
    if not spans:
        return None, "no_tracked_player"

    stats: Dict[str, Dict[str, int]] = {}
    for player, start, end in spans:
        token_len = end - start
        stat = stats.setdefault(player, {"best_len": 0, "count": 0, "first_start": start})
        stat["best_len"] = max(stat["best_len"], token_len)
        stat["count"] += 1
        stat["first_start"] = min(stat["first_start"], start)

    # Sort by: longest alias span, mention count, earliest position, alphabetical
    ordered = sorted(
        stats.items(),
        key=lambda kv: (-kv[1]["best_len"], -kv[1]["count"], kv[1]["first_start"], kv[0]),
    )

    reason = "longest_alias_then_count_then_position"
    if len(ordered) == 1:
        reason = "single_player"

    return ordered[0][0], reason


def sentence_record(sent, alias_to_player, matcher) -> Optional[Dict]:
    """Build a per-sentence record (players, primary_player, other_persons)."""
    sent_text = sent.text.strip()
    if not sent_text:
        return None

    sent_doc = sent.as_doc()
    spans: List[Tuple[str, int, int]] = []
    players_in_order: List[str] = []

    for _, start, end in matcher(sent_doc):
        span = sent_doc[start:end]
        alias_norm = normalize(span.text)
        player = alias_to_player.get(alias_norm)
        if not player:
            continue
        spans.append((player, start, end))
        if player not in players_in_order:
            players_in_order.append(player)

    other_persons: List[str] = []
    for ent in sent.ents:
        if ent.label_ != "PERSON":
            continue
        norm = normalize(ent.text)
        if norm in alias_to_player:
            continue
        if ent.text not in other_persons:
            other_persons.append(ent.text)

    if not players_in_order and not other_persons:
        return None

    primary_player, tie_breaker = pick_primary_player(spans)

    return {
        "sentence": sent_text,
        "players": players_in_order,
        "primary_player": primary_player,
        "tie_breaker": tie_breaker,
        "other_persons": other_persons,
    }


def process_url(
    url: str,
    nlp,
    matcher,
    alias_to_player,
    *,
    pronoun_backfill: bool = False,
    pronoun_fallback_player: Optional[str] = None,
) -> List[Dict]:
    text = fetch_text(url, suppress_errors=True)
    if not text:
        return []
    doc = nlp(text)
    rows: List[Dict] = []
    last_primary_player: Optional[str] = None
    for sent in doc.sents:
        rec = sentence_record(sent, alias_to_player, matcher)
        explicit_players = (rec or {}).get("players") or []

        if explicit_players:
            rows.append(rec)
            last_primary_player = rec.get("primary_player") or last_primary_player
            continue

        if not pronoun_backfill:
            if rec:
                rows.append(rec)
            continue

        if not sentence_has_pronoun(sent):
            if rec:
                rows.append(rec)
            continue

        context_player = last_primary_player
        if not context_player and pronoun_fallback_player:
            context_player = str(pronoun_fallback_player).strip() or None
        if not context_player:
            if rec:
                rows.append(rec)
            continue

        other_persons = (rec or {}).get("other_persons")
        rec_fill = build_pronoun_backfill_record(
            sent,
            alias_to_player,
            context_player,
            other_persons=other_persons,
        )
        if not rec_fill:
            if rec:
                rows.append(rec)
            continue

        rows.append(rec_fill)
        last_primary_player = context_player
    return rows


def run_batch(
    input_csv: Path,
    output_jsonl: Path,
    alias_csv: Path,
    model: str = "en_core_web_sm",
    max_urls: Optional[int] = None,
    pronoun_backfill: bool = False,
    pronoun_use_discovery_player_context: bool = True,
    assignment: str = "primary",
    store: str = "records",
    append_jsonl: bool = False,
    resume_jsonl: bool = False,
    flush_every: int = 1,
    write_url_markers: bool = True,
) -> int:
    if assignment not in {"all", "primary"}:
        raise ValueError("assignment must be one of: all, primary")
    if store not in {"lite", "records"}:
        raise ValueError("store must be one of: lite, records for JSONL output")

    alias_to_player, _ = load_alias_map(alias_csv)
    nlp = spacy.load(model)
    matcher = create_player_matcher(nlp, alias_to_player)

    df = pd.read_csv(input_csv)
    if "url" not in df.columns:
        raise ValueError("input CSV must contain a 'url' column")

    seen_urls = set()
    urls_attempted = 0
    written = 0

    done_urls: Set[str] = set()
    if resume_jsonl and output_jsonl.exists():
        done_urls = load_done_urls_from_jsonl(output_jsonl)

    mode = "a" if (append_jsonl or resume_jsonl) else "w"
    flush_every = max(1, int(flush_every))
    writes_since_flush = 0

    with output_jsonl.open(mode, encoding="utf-8") as outf:
        for _, row in df.iterrows():
            url = str(row.get("url", "")).strip()
            if not url or url in seen_urls or url in done_urls:
                continue
            if max_urls is not None and urls_attempted >= max_urls:
                break
            seen_urls.add(url)
            urls_attempted += 1

            meta = {
                "title": row.get("title", ""),
                "domain": row.get("domain", ""),
                "seendate": row.get("seendate", ""),
                "discovery_player": row.get("player", ""),
                "discovery_query": row.get("query", ""),
            }

            pronoun_fallback_player = (
                str(meta.get("discovery_player") or "").strip()
                if pronoun_use_discovery_player_context
                else None
            )

            try:
                rows = process_url(
                    url,
                    nlp,
                    matcher,
                    alias_to_player,
                    pronoun_backfill=pronoun_backfill,
                    pronoun_fallback_player=pronoun_fallback_player,
                )
            except Exception as exc:
                logging.warning("Failed %s: %s", url, exc)
                continue

            for rec in rows:
                rec_out = {"url": url, **meta, **rec}

                players = rec_out.get("players") or []
                primary = rec_out.get("primary_player")
                if assignment == "primary":
                    if not primary:
                        continue
                    assignment_players = [primary]
                else:
                    if not players:
                        continue
                    assignment_players = players

                for assigned_player in assignment_players:
                    if store == "lite":
                        out_obj = {
                            "player": assigned_player,
                            "url": url,
                            "sentence": rec_out.get("sentence", ""),
                            "primary_player": primary,
                            "players": players,
                            "other_persons": rec_out.get("other_persons") or [],
                            "tie_breaker": rec_out.get("tie_breaker"),
                        }
                    else:
                        out_obj = {"player": assigned_player, **rec_out}

                    outf.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    written += 1
                    writes_since_flush += 1
                    if writes_since_flush >= flush_every:
                        outf.flush()
                        writes_since_flush = 0

            if write_url_markers:
                outf.write(json.dumps({"_type": "url_done", "url": url}, ensure_ascii=False) + "\n")
                writes_since_flush += 1
                if writes_since_flush >= flush_every:
                    outf.flush()
                    writes_since_flush = 0
                if hasattr(os, "fsync"):
                    try:
                        os.fsync(outf.fileno())
                    except OSError:
                        pass

    return written


def run_batch_grouped_json(
    input_csv: Path,
    output_json: Path,
    alias_csv: Path,
    model: str = "en_core_web_sm",
    max_urls: Optional[int] = None,
    assignment: str = "all",
    store: str = "sentences",
    pronoun_backfill: bool = False,
    pronoun_use_discovery_player_context: bool = True,
    augment_existing: bool = False,
    augment_only_pronouns: bool = False,
    include_meta: bool = False,
    pretty: bool = True,
) -> Dict[str, object]:
    """Write grouped JSON where top-level keys are player names."""
    if assignment not in {"all", "primary"}:
        raise ValueError("assignment must be one of: all, primary")
    if store not in {"sentences", "lite", "records"}:
        raise ValueError("store must be one of: sentences, lite, records")

    alias_to_player, _ = load_alias_map(alias_csv)
    nlp = spacy.load(model)
    matcher = create_player_matcher(nlp, alias_to_player)

    df = pd.read_csv(input_csv)
    if "url" not in df.columns:
        raise ValueError("input CSV must contain a 'url' column")

    seen_urls = set()
    by_player: Dict[str, List[object]] = defaultdict(list)
    existing_sentence_keys: Dict[str, set[str]] = defaultdict(set)

    def sentence_key(value: object) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            s = value.get("sentence")
            return str(s).strip() if s is not None else ""
        return ""

    if augment_existing and output_json.exists():
        existing = json.loads(output_json.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise ValueError("existing output JSON must be an object (dict)")
        existing.pop("_meta", None)
        for player, items in existing.items():
            if not isinstance(items, list):
                continue
            by_player[player].extend(items)
            for item in items:
                key = sentence_key(item)
                if key:
                    existing_sentence_keys[player].add(key)

    urls_attempted = 0
    urls_with_sentences = 0
    sentence_records = 0
    assignments_written = 0

    for _, row in df.iterrows():
        url = str(row.get("url", "")).strip()
        if not url or url in seen_urls:
            continue
        if max_urls is not None and urls_attempted >= max_urls:
            break
        seen_urls.add(url)
        urls_attempted += 1

        meta = {
            "title": row.get("title", ""),
            "domain": row.get("domain", ""),
            "seendate": row.get("seendate", ""),
            "discovery_player": row.get("player", ""),
            "discovery_query": row.get("query", ""),
        }

        pronoun_fallback_player = (
            str(meta.get("discovery_player") or "").strip()
            if pronoun_use_discovery_player_context
            else None
        )

        try:
            rows = process_url(
                url,
                nlp,
                matcher,
                alias_to_player,
                pronoun_backfill=pronoun_backfill,
                pronoun_fallback_player=pronoun_fallback_player,
            )
        except Exception as exc:
            logging.warning("Failed %s: %s", url, exc)
            continue

        if not rows:
            continue
        urls_with_sentences += 1

        for rec in rows:
            sentence_records += 1
            rec_out = {"url": url, **meta, **rec}
            if augment_only_pronouns and rec_out.get("tie_breaker") != "pronoun_backfill":
                continue
            if store == "sentences":
                value: object = rec_out["sentence"]
            elif store == "lite":
                value = {
                    "sentence": rec_out["sentence"],
                    "primary_player": rec_out.get("primary_player"),
                    "players": rec_out.get("players") or [],
                    "other_persons": rec_out.get("other_persons") or [],
                }
            else:
                value = rec_out

            if assignment == "primary":
                primary = rec_out.get("primary_player")
                if primary:
                    if augment_existing:
                        key = sentence_key(value)
                        if key and key in existing_sentence_keys[primary]:
                            continue
                        existing_sentence_keys[primary].add(key)
                    by_player[primary].append(value)
                    assignments_written += 1
                continue

            players = rec_out.get("players") or []
            for player in players:
                if augment_existing:
                    key = sentence_key(value)
                    if key and key in existing_sentence_keys[player]:
                        continue
                    existing_sentence_keys[player].add(key)
                by_player[player].append(value)
                assignments_written += 1

    # Make output deterministic/easier to diff: sort players by name.
    by_player_sorted = dict(sorted(by_player.items(), key=lambda kv: kv[0].lower()))

    meta: Dict[str, object] = {
        "input_csv": str(input_csv),
        "alias_csv": str(alias_csv),
        "model": model,
        "assignment": assignment,
        "store": store,
        "pronoun_backfill": pronoun_backfill,
        "pronoun_use_discovery_player_context": pronoun_use_discovery_player_context,
        "augment_existing": augment_existing,
        "augment_only_pronouns": augment_only_pronouns,
        "unique_urls": len(seen_urls),
        "urls_attempted": urls_attempted,
        "urls_with_sentences": urls_with_sentences,
        "sentence_records": sentence_records,
        "assignments_written": assignments_written,
        "players_with_sentences": len(by_player_sorted),
    }

    out_payload: Dict[str, object] = dict(by_player_sorted)
    if include_meta:
        out_payload["_meta"] = meta

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as outf:
        json.dump(out_payload, outf, ensure_ascii=False, indent=2 if pretty else None)

    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Batch player sentence splitter; can emit JSONL per sentence or grouped JSON by player."
    )
    parser.add_argument(
        "--input-csv",
        default="data/media_data/discovered_urls_gdelt.csv",
        help="CSV with at least a 'url' column (e.g., discovered_urls_gdelt.csv)",
    )
    parser.add_argument(
        "--format",
        choices=["grouped-json", "jsonl"],
        default="grouped-json",
        help="Output format: grouped JSON by player, or sentence-level JSONL",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (defaults based on --format)",
    )
    parser.add_argument(
        "--alias-csv",
        default="data/player_data/player_aliases.csv",
        help="Path to player alias CSV with columns: alias, player",
    )
    parser.add_argument(
        "--model",
        default="en_core_web_sm",
        help="spaCy model to load",
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=None,
        help="Optional cap on number of unique URLs to process",
    )
    parser.add_argument(
        "--assignment",
        choices=["all", "primary"],
        default="all",
        help="For grouped JSON: store each sentence under all matched players or only the primary player",
    )
    parser.add_argument(
        "--store",
        choices=["sentences", "lite", "records"],
        default="sentences",
        help="For grouped JSON: store just sentence strings, a small per-sentence dict, or full per-sentence records",
    )
    parser.add_argument(
        "--pretty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pretty-print JSON output (grouped-json only; larger files). Use --no-pretty for compact JSON.",
    )
    parser.add_argument(
        "--include-meta",
        action="store_true",
        help="Include a top-level '_meta' key with run stats (grouped-json only)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing output file even if it looks like the wrong format",
    )
    parser.add_argument(
        "--pronoun-backfill",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Heuristic: assign pronoun-only sentences (he/him/his) to a context player.",
    )
    parser.add_argument(
        "--pronoun-use-discovery-player-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When pronoun backfill is on, fall back to the CSV's discovery player if no in-article player was seen.",
    )
    parser.add_argument(
        "--augment-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If output exists (grouped-json), load it and append new records instead of overwriting.",
    )
    parser.add_argument(
        "--augment-only-pronouns",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When augmenting, only append sentences created by the pronoun heuristic.",
    )
    parser.add_argument(
        "--append-jsonl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For JSONL output: append to the output file instead of overwriting it.",
    )
    parser.add_argument(
        "--resume-jsonl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For JSONL output: skip URLs already present in the output file (implies append).",
    )
    parser.add_argument(
        "--jsonl-flush-every",
        type=int,
        default=1,
        help="For JSONL output: flush every N written lines (smaller is safer, larger is faster).",
    )
    parser.add_argument(
        "--jsonl-write-url-markers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For JSONL output: write {'_type':'url_done','url':...} marker lines to support safer resume.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    input_csv = Path(args.input_csv)
    alias_csv = Path(args.alias_csv)

    if args.format == "jsonl":
        output_path = Path(args.output) if args.output else Path("data/media_data/player_sentences.jsonl")
        assert_safe_output_path(args.format, output_path, args.force)

        store_jsonl = args.store
        if store_jsonl == "sentences":
            store_jsonl = "records"  # backward compatible default for JSONL

        written = run_batch(
            input_csv=input_csv,
            output_jsonl=output_path,
            alias_csv=alias_csv,
            model=args.model,
            max_urls=args.max_urls,
            pronoun_backfill=args.pronoun_backfill,
            pronoun_use_discovery_player_context=args.pronoun_use_discovery_player_context,
            assignment=args.assignment,
            store=store_jsonl,
            append_jsonl=args.append_jsonl or args.resume_jsonl,
            resume_jsonl=args.resume_jsonl,
            flush_every=args.jsonl_flush_every,
            write_url_markers=args.jsonl_write_url_markers,
        )
        logging.info("Wrote %d JSONL records to %s", written, output_path)
        return

    output_path = (
        Path(args.output) if args.output else Path("data/media_data/player_sentences_by_player.json")
    )
    assert_safe_output_path(args.format, output_path, args.force)
    meta = run_batch_grouped_json(
        input_csv=input_csv,
        output_json=output_path,
        alias_csv=alias_csv,
        model=args.model,
        max_urls=args.max_urls,
        assignment=args.assignment,
        store=args.store,
        pronoun_backfill=args.pronoun_backfill,
        pronoun_use_discovery_player_context=args.pronoun_use_discovery_player_context,
        augment_existing=args.augment_existing,
        augment_only_pronouns=args.augment_only_pronouns,
        include_meta=args.include_meta,
        pretty=args.pretty,
    )
    logging.info(
        "Wrote grouped JSON to %s (urls_attempted=%s, assignments_written=%s, players_with_sentences=%s)",
        output_path,
        meta.get("urls_attempted"),
        meta.get("assignments_written"),
        meta.get("players_with_sentences"),
    )


if __name__ == "__main__":
    main()
