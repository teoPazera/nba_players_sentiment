"""Convert grouped-by-player JSON into JSON Lines (JSONL)."""

import argparse
import json
from pathlib import Path
from json import JSONDecodeError
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


def iter_grouped_items(payload: Dict) -> Iterator[Tuple[str, object]]:
    for player, items in payload.items():
        if player == "_meta":
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            yield str(player), item


def sentence_key(player: str, item: object) -> Optional[str]:
    if isinstance(item, str):
        sent = item.strip()
        return f"{player}\t{sent}" if sent else None
    if isinstance(item, dict):
        sent = item.get("sentence")
        sent_str = str(sent).strip() if sent is not None else ""
        return f"{player}\t{sent_str}" if sent_str else None
    return None


def convert_grouped_json_to_jsonl(
    input_json: Path,
    output_jsonl: Path,
    *,
    append: bool = False,
    dedupe: bool = True,
    flush_every: int = 1000,
) -> int:
    try:
        payload = json.loads(input_json.read_text(encoding="utf-8-sig"))
    except JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse {input_json} as grouped JSON. "
            "If this file was accidentally written as JSONL, restore the grouped JSON backup and re-run."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("input JSON must be an object (dict) keyed by player")

    mode = "a" if append else "w"
    flush_every = max(1, int(flush_every))
    seen: set[str] = set()
    written = 0

    with output_jsonl.open(mode, encoding="utf-8") as outf:
        for player, item in iter_grouped_items(payload):
            if isinstance(item, str):
                rec: Dict[str, object] = {"player": player, "sentence": item}
            elif isinstance(item, dict):
                rec = {"player": player, **item}
            else:
                continue

            if dedupe:
                key = sentence_key(player, item)
                if key and key in seen:
                    continue
                if key:
                    seen.add(key)

            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if written % flush_every == 0:
                outf.flush()

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert grouped player JSON to JSONL.")
    parser.add_argument(
        "--input-json",
        default="data/media_data/player_sentences_by_player.json",
        help="Grouped JSON by player (from scripts/article_player_batch.py)",
    )
    parser.add_argument(
        "--output-jsonl",
        default="data/media_data/player_sentences_by_player.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--append",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append to output JSONL instead of overwriting",
    )
    parser.add_argument(
        "--dedupe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dedupe by (player, sentence) while converting",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1000,
        help="Flush every N lines",
    )
    args = parser.parse_args()

    written = convert_grouped_json_to_jsonl(
        input_json=Path(args.input_json),
        output_jsonl=Path(args.output_jsonl),
        append=args.append,
        dedupe=args.dedupe,
        flush_every=args.flush_every,
    )

    print(f"Wrote {written} JSONL records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
