"""Convert scraped Reddit JSON into sentence-level data grouped by player."""

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import spacy

from article_player_batch import sentence_record
from article_player_split import create_player_matcher, load_alias_map
from pronoun_backfill import build_pronoun_backfill_record, sentence_has_pronoun


def iter_reddit_text_blocks(
    posts: List[Dict],
    include_comments: bool = True,
    max_posts: Optional[int] = None,
) -> Iterable[Tuple[str, str, Dict]]:
    """Yield (source_type, text, meta) blocks for post title/body and comments."""
    yielded_posts = 0
    for post in posts:
        if max_posts is not None and yielded_posts >= max_posts:
            break
        yielded_posts += 1

        base_meta = {
            "source": "reddit",
            "search_player": post.get("search_player", ""),
            "subreddit": post.get("subreddit", ""),
            "post_id": post.get("post_id", ""),
            "post_url": post.get("post_url", ""),
            "created_utc": post.get("created_utc", None),
            "created_date_utc": post.get("created_date_utc", ""),
        }

        title = str(post.get("title") or "").strip()
        if title:
            yield "post_title", title, base_meta

        body = str(post.get("selftext") or "").strip()
        if body:
            yield "post_body", body, base_meta

        if not include_comments:
            continue

        comments = post.get("comments") or []
        if not isinstance(comments, list):
            continue

        for comment in comments:
            if not isinstance(comment, dict):
                continue
            c_body = str(comment.get("body") or "").strip()
            if not c_body:
                continue
            meta = dict(base_meta)
            meta.update(
                {
                    "comment_id": comment.get("comment_id", ""),
                    "parent_id": comment.get("parent_id", ""),
                    "comment_created_utc": comment.get("created_utc", None),
                    "comment_created_date_utc": comment.get("created_date_utc", ""),
                }
            )
            yield "comment", c_body, meta


def run_sentence_batch(
    input_json: Path,
    output_json: Path,
    alias_csv: Path,
    model: str = "en_core_web_sm",
    assignment: str = "all",
    store: str = "lite",
    include_comments: bool = True,
    pronoun_backfill: bool = True,
    pronoun_use_search_player_context: bool = True,
    include_meta: bool = False,
    pretty: bool = True,
    max_posts: Optional[int] = None,
) -> Dict[str, object]:
    if assignment not in {"all", "primary"}:
        raise ValueError("assignment must be one of: all, primary")
    if store not in {"sentences", "lite", "records"}:
        raise ValueError("store must be one of: sentences, lite, records")

    posts = json.loads(input_json.read_text(encoding="utf-8"))
    if not isinstance(posts, list):
        raise ValueError("input JSON must be a list of post records")

    alias_to_player, _ = load_alias_map(alias_csv)
    nlp = spacy.load(model)
    matcher = create_player_matcher(nlp, alias_to_player)

    by_player: Dict[str, List[object]] = defaultdict(list)

    blocks_seen = 0
    sentence_records = 0
    assignments_written = 0
    pronoun_backfilled_sentences = 0

    for source_type, text, meta in iter_reddit_text_blocks(
        posts, include_comments=include_comments, max_posts=max_posts
    ):
        blocks_seen += 1
        last_primary_player: Optional[str] = None
        doc = nlp(text)
        for sent in doc.sents:
            rec = sentence_record(sent, alias_to_player, matcher)
            explicit_players = (rec or {}).get("players") or []

            if explicit_players:
                sentence_records += 1
                rec_out = {"source_type": source_type, **meta, **rec}
                last_primary_player = rec_out.get("primary_player") or last_primary_player
            else:
                if not pronoun_backfill:
                    continue
                if not sentence_has_pronoun(sent):
                    continue

                context_player = last_primary_player
                if not context_player and pronoun_use_search_player_context:
                    context_player = str(meta.get("search_player") or "").strip() or None
                if not context_player:
                    continue

                other_persons = (rec or {}).get("other_persons")
                rec_fill = build_pronoun_backfill_record(
                    sent,
                    alias_to_player,
                    context_player,
                    other_persons=other_persons,
                )
                if not rec_fill:
                    continue

                sentence_records += 1
                pronoun_backfilled_sentences += 1
                rec_out = {"source_type": source_type, **meta, **rec_fill}
                last_primary_player = context_player

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
                    by_player[primary].append(value)
                    assignments_written += 1
                continue

            players = rec_out.get("players") or []
            for player in players:
                by_player[player].append(value)
                assignments_written += 1

    by_player_sorted = dict(sorted(by_player.items(), key=lambda kv: kv[0].lower()))

    meta_out: Dict[str, object] = {
        "input_json": str(input_json),
        "output_json": str(output_json),
        "alias_csv": str(alias_csv),
        "model": model,
        "assignment": assignment,
        "store": store,
        "include_comments": include_comments,
        "max_posts": max_posts,
        "posts_in_input": len(posts),
        "text_blocks_seen": blocks_seen,
        "sentence_records": sentence_records,
        "pronoun_backfilled_sentences": pronoun_backfilled_sentences,
        "assignments_written": assignments_written,
        "players_with_sentences": len(by_player_sorted),
    }

    out_payload: Dict[str, object] = dict(by_player_sorted)
    if include_meta:
        out_payload["_meta"] = meta_out

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as outf:
        json.dump(out_payload, outf, ensure_ascii=False, indent=2 if pretty else None)

    return meta_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split scraped Reddit posts/comments into sentences and group them by player."
    )
    parser.add_argument(
        "--input-json",
        default="data/reddit_data/reddit_player_posts_may_sep_2025_50-50-1-30-1.json",
        help="Path to JSON produced by reddit_scraper.py",
    )
    parser.add_argument(
        "--output-json",
        default="data/reddit_data/reddit_player_sentences_by_player.json",
        help="Where to write grouped JSON by player",
    )
    parser.add_argument(
        "--alias-csv",
        default="data/player_data/player_aliases.csv",
        help="Path to player alias CSV with columns: alias, player",
    )
    parser.add_argument("--model", default="en_core_web_sm", help="spaCy model to load")
    parser.add_argument(
        "--assignment",
        choices=["all", "primary"],
        default="primary",
        help="Store each sentence under all matched players or only the primary player",
    )
    parser.add_argument(
        "--store",
        choices=["sentences", "lite", "records"],
        default="lite",
        help="Store sentence strings, a small per-sentence dict, or full records",
    )
    parser.add_argument(
        "--include-comments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include comment bodies as well as post title/selftext",
    )
    parser.add_argument(
        "--pronoun-backfill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Heuristic: assign pronoun-only sentences (he/him/his) to a context player.",
    )
    parser.add_argument(
        "--pronoun-use-search-player-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When pronoun backfill is on, fall back to the post's search_player if no in-block player was seen.",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=None,
        help="Optional cap on number of posts to process (for quick testing)",
    )
    parser.add_argument(
        "--pretty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pretty-print JSON output. Use --no-pretty for compact JSON.",
    )
    parser.add_argument(
        "--include-meta",
        action="store_true",
        help="Include a top-level '_meta' key with run stats",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    meta = run_sentence_batch(
        input_json=Path(args.input_json),
        output_json=Path(args.output_json),
        alias_csv=Path(args.alias_csv),
        model=args.model,
        assignment=args.assignment,
        store=args.store,
        include_comments=args.include_comments,
        pronoun_backfill=args.pronoun_backfill,
        pronoun_use_search_player_context=args.pronoun_use_search_player_context,
        include_meta=args.include_meta,
        pretty=args.pretty,
        max_posts=args.max_posts,
    )

    logging.info(
        "Wrote %s (sentence_records=%s, assignments_written=%s, players_with_sentences=%s)",
        args.output_json,
        meta.get("sentence_records"),
        meta.get("assignments_written"),
        meta.get("players_with_sentences"),
    )


if __name__ == "__main__":
    main()
