# Do Fans and Media Move the Market?
## 2025 NBA free-agency project: Reddit + media sentiment vs player salary

This repo supports a research project asking:

> Whether (and to what extent) pre-signing sentiment from (1) NBA fans on Reddit and (2) sports media coverage is associated with the salary a free agent receives, after controlling for on-court performance.

The project is built as a pipeline: assemble a player-level dataset, collect text in a consistent pre-signing window, attribute sentences to players, score sentiment, then test whether sentiment adds explanatory power beyond performance.

---

## Story/pipeline

### 1) Player dataset (who signed, when, and for how much)
Goal: a clean table of 2025 offseason signings with:
- salary outcome: AAV when possible; otherwise a fallback proxy (e.g., next-season salary)
- signing date (used to end each player's text-collection window)
- last-season performance controls (we ultimately use Win Shares (WS) as the baseline predictor)

Relevant inputs live in `data/player_data/` (signings lists, salary tables, stats exports).

### 2) Aliases + nicknames (to improve entity linking)
To match players in text more reliably (names, unique surnames, nicknames), we build an alias map:
- `wiki_scrapper.py` -> writes `data/player_data/player_aliases.csv`

This alias file is used by both the Reddit and media pipelines.

### 3) Reddit collection (fans)
We collect pre-signing discussion from three subreddits:
- `r/nba`, `r/nbadiscussion`, `r/NBAtalk`

Rules (as implemented in `reddit_scraper.py`):
- per-player time window: from `2025-05-01` up to that player's signing date
- search query: OR over the player's aliases
- per (player, subreddit): up to 200 matching posts
- filter noisy posts: cap total PERSON entities in the post at 10
- comments: keep up to 40 comments per post, only if PERSON entities in the comment are <= 5

Output:
- `data/reddit_data/reddit_player_posts_*.json`

### 4) Reddit -> sentence dataset
We split posts and comments into sentences and assign them to players:
- `scripts/reddit_player_sentence_batch.py` -> grouped JSON/JSONL by player

Attribution uses:
- alias matching (spaCy PhraseMatcher over `player_aliases.csv`)
- a lightweight pronoun backfill heuristic for pronoun-only sentences (optional)

Outputs:
- `data/reddit_data/reddit_player_sentences_by_player_*.json`
- `data/reddit_data/reddit_player_sentences_by_player_*.jsonl`

### 5) Media discovery + scraping (coverage)
We use the GDELT Doc API to discover candidate URLs per player:
- `media_discover_gdelt.py` -> `data/media_data/discovered_urls_gdelt.csv`

Then we scrape each URL and split into player-attributed sentences:
- `scripts/article_player_batch.py`
- optional helper: `scripts/grouped_player_json_to_jsonl.py`

Outputs:
- `data/media_data/player_sentences_by_player.json` (grouped JSON)
- `data/media_data/player_sentences_by_player.jsonl` (JSONL)

Media is noisier than Reddit (context is not guaranteed to be NBA/free-agency), so sentence attribution and pronoun backfill matter more here.

### 6) Sentiment scoring
We score sentences with:
- VADER (lexicon-based; good baseline for social/media text)
- optional DeBERTa ABSA (target-aware; "sentiment toward this player")

Code:
- `classifier.py`
- `scripts/sentiment_pipeline.py` -> writes JSONL incrementally (supports resume/dedupe)

Outputs:
- `data/sentiment_data/reddit_sentence_sentiment.jsonl`
- `data/sentiment_data/media_sentence_sentiment.jsonl`

### 7) EDA + salary analysis
Notebooks:
- `eda_sentence_sentiment.ipynb` (distribution checks, tie-breakers, attribution noise, etc.)
- `sentiment_overpay_analysis.ipynb`

Modeling idea used in the analysis notebook:
- baseline: `log(AAV) ~ WS`
- define "overpay/underpay" as the residual from the baseline model
- test whether aggregated (player-level) sentiment explains residual salary beyond WS

---

## Setup

### 1) Python environment
Install dependencies:
- `pip install -r requirements.txt`

Install spaCy model:
- `python -m spacy download en_core_web_sm`

### 2) Reddit credentials
`reddit_scraper.py` and `reddit_test.py` read credentials from environment variables:
- `REDDIT_CLIENT_ID`
- `REDDIT_CLIENT_SECRET`
- optional: `REDDIT_USER_AGENT` (default: `nba-sentiment-research`)

Copy `.env.example` to `.env` and fill values. Do not commit `.env`.

### 3) DeBERTa (optional)
If you enable DeBERTa scoring in `scripts/sentiment_pipeline.py`, the model may need to be downloaded once.
If you cannot download (restricted network), run with `--no-deberta` or `--deberta-local-files-only` (requires a pre-populated HF cache).

---

## Reproduce (high level)

1. Build aliases: `python wiki_scrapper.py`
2. Reddit scrape: `python reddit_scraper.py`
3. Reddit -> sentences: `python scripts/reddit_player_sentence_batch.py`
4. Discover media URLs: `python media_discover_gdelt.py`
5. Media -> sentences: `python scripts/article_player_batch.py`
6. Sentiment scoring: `python scripts/sentiment_pipeline.py`
7. Run notebooks for EDA + analysis

---

## Notes / limitations (brief)
- Pronoun backfill is a heuristic (not true coreference) and can misattribute sentences in multi-person contexts.
- Media context is noisier than Reddit; filtering and attribution choices matter.
- Sentence counts are highly imbalanced by player; player-level aggregation and minimum-sentence robustness checks help.
