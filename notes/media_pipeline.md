# Media Pipeline Notes

- Discovery script: `media_discover_gdelt.py` (GDELT Doc API).
- Current queries: per-player aliases combined with short market/value/opinion/free-agency terms.
- Deduplication: URLs are de-duplicated across players/queries within one run (`seen_urls` set), so the same article is only written once.
- Output: `data/media_data/discovered_urls_gdelt.csv` (or `--out-csv` path) with columns like `url`, `title`, `domain`, `seendate`, and the query used.
- Running for all players:  
  `.venv\Scripts\python.exe media_discover_gdelt.py --max-records 50 --out-csv data/media_data/discovered_urls_gdelt.csv`
- To test a subset: use `--max-players N` or `--player "Name"`.
- To rerun only players with a single alias (helps avoid GDELT parenthesis errors on single-term queries): add `--single-alias-only`.
- To append to an existing output instead of overwriting (default behavior), leave `--append` on; it will load existing URLs and dedupe by `url`. Use `--no-append` to overwrite.

## Linking statements to players
- Sentence splitter (`scripts/article_player_split.py`): loads `player_aliases.csv`, builds a spaCy `PhraseMatcher` over aliases, fetches an article, extracts main text with `trafilatura` (falls back to raw HTML text), tokenizes into sentences, and attaches each sentence to tracked players whose aliases appear. Other PERSON entities from spaCy NER are listed separately. Sentences with no tracked players and no PERSON entities are dropped.
- Output shape (per sentence): `sentence`, `players` (semicolon-joined deduped player names in sentence order), `other_persons` (other PERSON strings). The CLI prints each record; `process_url` returns a list of dicts.
- Batch exporter (`scripts/article_player_batch.py`): runs the same matching over many URLs from a CSV (e.g., `data/media_data/discovered_urls_gdelt.csv`) and writes:
  - grouped JSON by player (default): `data/media_data/player_sentences_by_player.json` with top-level keys as player names and values as sentence lists
  - sentence-level JSONL (optional): `--format jsonl` for one JSON record per sentence (useful for streaming / big datasets)
- Typical run (grouped JSON of sentence strings):
  `.venv\Scripts\python.exe scripts/article_player_batch.py --input-csv data/media_data/discovered_urls_gdelt.csv --max-urls 200`
- If you want *just* sentence + player/person info (no URL/title metadata): add `--store lite`.
- If you want URL/title metadata per sentence: add `--store records`.
- If you want run stats written into the JSON: add `--include-meta` (writes a top-level `_meta` key).
- Output JSON is pretty-printed by default; use `--no-pretty` for compact output.
- Pronoun heuristic (not true coreference): add `--pronoun-backfill` to assign pronoun-only sentences (he/him/his) to a context player. By default it falls back to the discovery CSV's `player` when no in-article player was seen; disable that with `--no-pronoun-use-discovery-player-context`.
- To *extend* an existing grouped JSON file (instead of overwriting): add `--augment-existing` (optionally `--augment-only-pronouns` to append only pronoun-backfilled sentences).
- To make long runs resilient to failures: use `--format jsonl` with `--store lite` and `--resume-jsonl` (writes as it goes and skips URLs already completed).
- To convert an existing grouped JSON into JSONL for easier incremental workflows: `scripts/grouped_player_json_to_jsonl.py`.
- Alias hygiene: keep full names in `player_aliases.csv`; avoid ambiguous last names unless they are unique. For shared surnames (e.g., Alexander-Walker), prefer multi-token aliases and drop bare last names or explicitly flag them to avoid collisions.
- Future refinements: longest-match-wins (favor full names over substrings), coreference/pronoun attachment, and sentence-level confidence scoring for downstream sentiment.
- Player column in discovery CSVs is only for traceability of which query found the URL; final attribution should come from in-text alias matching.

## Reddit pipeline (posts/comments -> player sentences)
- Scraper (`reddit_scraper.py`): collects Reddit posts (and selected comments) per player into a JSON file under `data/reddit_data/`.
- Sentence batch (`scripts/reddit_player_sentence_batch.py`): reads that scraped JSON, splits post titles/selftext/comments into sentences, matches tracked players by alias, and writes grouped JSON by player.
- Typical run (primary-player assignment, lite records, pretty JSON):
  `.venv\Scripts\python.exe scripts/reddit_player_sentence_batch.py --input-json data/reddit_data/reddit_player_posts_may_sep_2025_50-50-1-30-1.json`
- Pronoun heuristic (not true coreference): the sentence batch can optionally assign pronoun-only sentences (he/him/his) to a context player; disable with `--no-pronoun-backfill`.

## Sentiment scoring (media + reddit)
- Pipeline script: `scripts/sentiment_pipeline.py`
- Inputs: sentence datasets as JSONL or grouped JSON (from the media/reddit sentence batch scripts).
- Outputs (JSONL, written incrementally):  
  - `data/sentiment_data/media_sentence_sentiment.jsonl`  
  - `data/sentiment_data/reddit_sentence_sentiment.jsonl`
- Each output record is compact and includes: `player`, `other_players`, `n_other_players`, VADER (`vader`, `vader_label`) and DeBERTa ABSA (`deberta`, `deberta_label`).
  - By default it also includes per-class scores: `vader_neg/neu/pos` and `deberta_neg/neu/pos` (disable with `--no-include-probs`).
- Quick test (score a few records from each source, using local model cache only):
  `.venv\Scripts\python.exe scripts/sentiment_pipeline.py --max-media 20 --max-reddit 20 --deberta-local-files-only`
- Resume behavior: `--resume` is on by default and dedupes by `(source, player, sentence)` so re-running appends only new records.
