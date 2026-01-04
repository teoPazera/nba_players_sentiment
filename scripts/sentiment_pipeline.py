"""Score sentence datasets with VADER and optional DeBERTa ABSA.

Reads grouped JSON or JSONL and writes JSONL outputs (supports resume/dedupe).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

# Ensure repo root is importable when running `python scripts/...`
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from article_player_split import load_alias_map
from classifier import DebertaABSAScorer, vader_score_sentence


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if ";" in s:
            return [p.strip() for p in s.split(";") if p.strip()]
        return [s]
    return [value]


def _record_key(source: str, player: str, sentence: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(source.encode("utf-8", errors="ignore"))
    h.update(b"\0")
    h.update(player.encode("utf-8", errors="ignore"))
    h.update(b"\0")
    h.update(sentence.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def iter_jsonl_records(path: Path, *, source: str) -> Iterator[Dict[str, Any]]:
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
            if obj.get("_type") == "url_done":
                continue
            obj = dict(obj)
            obj["_source"] = source
            obj["_input_path"] = str(path)
            yield obj


def iter_grouped_json_records(path: Path, *, source: str) -> Iterator[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected grouped JSON object (dict) in {path}")

    for player, items in payload.items():
        if player == "_meta":
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, str):
                rec: Dict[str, Any] = {"player": str(player), "sentence": item}
            elif isinstance(item, dict):
                rec = {"player": str(player), **item}
            else:
                continue
            rec["_source"] = source
            rec["_input_path"] = str(path)
            yield rec


def iter_input_records(paths: Iterable[Path], *, source: str) -> Iterator[Dict[str, Any]]:
    for path in paths:
        if not path.exists():
            logging.warning("Input does not exist, skipping: %s", path)
            continue
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            yield from iter_jsonl_records(path, source=source)
        elif suffix == ".json":
            yield from iter_grouped_json_records(path, source=source)
        else:
            raise ValueError(f"Unsupported input type: {path}")


def load_seen_keys_from_output(output_jsonl: Path, *, source: str) -> Set[str]:
    seen: Set[str] = set()
    try:
        with output_jsonl.open("r", encoding="utf-8") as f:
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
                player = str(obj.get("player") or "").strip()
                sentence = str(obj.get("sentence") or "").strip()
                if not player or not sentence:
                    continue
                seen.add(_record_key(source, player, sentence))
    except FileNotFoundError:
        return set()
    except OSError:
        return set()
    return seen


def _safe_open_output(path: Path, *, resume: bool, force: bool):
    if path.exists() and not resume and not force:
        raise ValueError(f"Output exists: {path}. Use --resume to append or --force to overwrite.")
    mode = "a" if resume else "w"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open(mode, encoding="utf-8")


def score_dataset(
    *,
    source: str,
    inputs: List[Path],
    output: Path,
    max_records: Optional[int],
    resume: bool,
    force: bool,
    flush_every: int,
    dedupe: bool,
    use_vader: bool,
    deberta: Optional[DebertaABSAScorer],
    include_probs: bool,
) -> Dict[str, Any]:
    flush_every = max(1, int(flush_every))

    seen_keys: Set[str] = set()
    if dedupe and resume and output.exists():
        seen_keys = load_seen_keys_from_output(output, source=source)
        logging.info("Resume enabled: loaded %d existing keys from %s", len(seen_keys), output)

    read_records = 0
    scored_records = 0
    skipped_existing = 0
    written = 0

    with _safe_open_output(output, resume=resume, force=force) as outf:
        for rec in iter_input_records(inputs, source=source):
            read_records += 1

            player = str(rec.get("player") or rec.get("primary_player") or "").strip()
            sentence = str(rec.get("sentence") or "").strip()
            if not player or not sentence:
                continue

            if dedupe:
                k = _record_key(source, player, sentence)
                if k in seen_keys:
                    skipped_existing += 1
                    continue
                seen_keys.add(k)

            players_list = [str(p).strip() for p in _ensure_list(rec.get("players")) if str(p).strip()]
            other_players = sorted({p for p in players_list if p and p != player})

            vader_out = vader_score_sentence(sentence) if use_vader else None
            deberta_out = (
                deberta.score_sentence(sentence, primary_player=player, resolved_player=player)
                if deberta
                else None
            )

            vader_compound = None
            vader_label = None
            vader_neg = None
            vader_neu = None
            vader_pos = None
            if vader_out and isinstance(vader_out, dict):
                vader_scores = vader_out.get("scores") or {}
                vader_compound = vader_scores.get("compound", vader_out.get("value"))
                vader_label = vader_out.get("label")
                if include_probs:
                    vader_neg = vader_scores.get("neg")
                    vader_neu = vader_scores.get("neu")
                    vader_pos = vader_scores.get("pos")

            deberta_value = None
            deberta_label = None
            deberta_neg = None
            deberta_neu = None
            deberta_pos = None
            if deberta_out and isinstance(deberta_out, dict):
                deberta_value = deberta_out.get("value")
                deberta_label = deberta_out.get("label")
                if include_probs:
                    deberta_scores = deberta_out.get("scores") or {}
                    deberta_neg = deberta_scores.get("Negative")
                    deberta_neu = deberta_scores.get("Neutral")
                    deberta_pos = deberta_scores.get("Positive")

            out_obj: Dict[str, Any] = {
                "source": source,
                "player": player,
                "sentence": sentence,
                "other_players": other_players,
                "n_other_players": len(other_players),
                "vader": vader_compound,
                "vader_label": vader_label,
                "deberta": deberta_value,
                "deberta_label": deberta_label,
            }
            if include_probs:
                out_obj["vader_neg"] = vader_neg
                out_obj["vader_neu"] = vader_neu
                out_obj["vader_pos"] = vader_pos
                out_obj["deberta_neg"] = deberta_neg
                out_obj["deberta_neu"] = deberta_neu
                out_obj["deberta_pos"] = deberta_pos

            tie_breaker = rec.get("tie_breaker")
            if tie_breaker not in (None, ""):
                out_obj["tie_breaker"] = tie_breaker

            outf.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            written += 1
            scored_records += 1

            if written % flush_every == 0:
                outf.flush()

            if max_records is not None and scored_records >= max_records:
                break

    return {
        "source": source,
        "inputs": [str(p) for p in inputs],
        "output": str(output),
        "read_records": read_records,
        "scored_records": scored_records,
        "written": written,
        "skipped_existing": skipped_existing,
        "dedupe": dedupe,
        "resume": resume,
        "use_vader": use_vader,
        "use_deberta": bool(deberta),
        "deberta_model_id": getattr(deberta, "model_id", None) if deberta else None,
    }


def _default_existing(paths: List[Path]) -> List[Path]:
    return [p for p in paths if p.exists()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score media + reddit sentence datasets with VADER + DeBERTa ABSA and write JSONL outputs."
    )
    parser.add_argument(
        "--media-input",
        action="append",
        default=None,
        help="Media input file (.jsonl or grouped .json). Can be passed multiple times.",
    )
    parser.add_argument(
        "--reddit-input",
        action="append",
        default=None,
        help="Reddit input file (.jsonl or grouped .json). Can be passed multiple times.",
    )
    parser.add_argument(
        "--media-output",
        default="data/sentiment_data/media_sentence_sentiment.jsonl",
        help="Media output JSONL path",
    )
    parser.add_argument(
        "--reddit-output",
        default="data/sentiment_data/reddit_sentence_sentiment.jsonl",
        help="Reddit output JSONL path",
    )
    parser.add_argument(
        "--alias-csv",
        default="data/player_data/player_aliases.csv",
        help="Alias CSV (player,alias) used by DeBERTa ABSA mention detection",
    )
    parser.add_argument(
        "--max-media",
        type=int,
        default=None,
        help="Optional cap on #scored media sentence records (for quick testing)",
    )
    parser.add_argument(
        "--max-reddit",
        type=int,
        default=None,
        help="Optional cap on #scored reddit sentence records (for quick testing)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append to outputs and skip already-scored (player,sentence) pairs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting outputs when --no-resume is used",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Flush outputs every N lines (smaller is safer, larger is faster)",
    )
    parser.add_argument(
        "--dedupe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dedupe by (source, player, sentence) while scoring",
    )
    parser.add_argument(
        "--vader",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable VADER scoring",
    )
    parser.add_argument(
        "--deberta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable DeBERTa ABSA scoring",
    )
    parser.add_argument(
        "--deberta-model-id",
        default="yangheng/deberta-v3-base-absa-v1.1",
        help="HuggingFace model id for DeBERTa ABSA",
    )
    parser.add_argument(
        "--deberta-device",
        type=int,
        default=None,
        help="Device for transformers pipeline (-1 CPU, 0 GPU). Default: auto",
    )
    parser.add_argument(
        "--deberta-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require DeBERTa model/tokenizer to be present in local HF cache (no downloads).",
    )
    parser.add_argument(
        "--allow-missing-deberta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If DeBERTa can't be loaded, keep going and write only VADER scores.",
    )
    parser.add_argument(
        "--include-probs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include per-class (neg/neu/pos) scores for VADER and DeBERTa (adds 6 float columns).",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    media_inputs = (
        [Path(p) for p in args.media_input]
        if args.media_input
        else _default_existing(
            [
                Path("data/media_data/media_pronoun_run.jsonl"),
                Path("data/media_data/player_sentences_by_player.jsonl"),
            ]
        )
    )
    reddit_inputs = (
        [Path(p) for p in args.reddit_input]
        if args.reddit_input
        else _default_existing(
            [
                Path("data/reddit_data/reddit_player_sentences_by_player_lite_pronouns.jsonl"),
                Path("data/reddit_data/reddit_player_sentences_by_player_lite_pronouns.json"),
            ]
        )
    )

    alias_csv = Path(args.alias_csv)

    deberta_scorer: Optional[DebertaABSAScorer] = None
    if args.deberta:
        try:
            _, player_to_aliases = load_alias_map(alias_csv)
            player_aliases = {p: sorted(list(a)) for p, a in player_to_aliases.items()}
            deberta_scorer = DebertaABSAScorer(
                player_aliases,
                model_id=args.deberta_model_id,
                device=args.deberta_device,
                local_files_only=args.deberta_local_files_only,
            )
        except Exception as exc:
            if not args.allow_missing_deberta:
                raise
            logging.warning("DeBERTa scorer unavailable (%s). Continuing without it.", exc)
            deberta_scorer = None

    results: List[Dict[str, Any]] = []
    if media_inputs:
        results.append(
            score_dataset(
                source="media",
                inputs=media_inputs,
                output=Path(args.media_output),
                max_records=args.max_media,
                resume=args.resume,
                force=args.force,
                flush_every=args.flush_every,
                dedupe=args.dedupe,
                use_vader=args.vader,
                deberta=deberta_scorer,
                include_probs=args.include_probs,
            )
        )
    else:
        logging.warning("No media inputs provided/found; skipping media scoring.")

    if reddit_inputs:
        results.append(
            score_dataset(
                source="reddit",
                inputs=reddit_inputs,
                output=Path(args.reddit_output),
                max_records=args.max_reddit,
                resume=args.resume,
                force=args.force,
                flush_every=args.flush_every,
                dedupe=args.dedupe,
                use_vader=args.vader,
                deberta=deberta_scorer,
                include_probs=args.include_probs,
            )
        )
    else:
        logging.warning("No reddit inputs provided/found; skipping reddit scoring.")

    for r in results:
        logging.info(
            "Done %s: wrote=%s skipped_existing=%s output=%s",
            r.get("source"),
            r.get("written"),
            r.get("skipped_existing"),
            r.get("output"),
        )


if __name__ == "__main__":
    main()
