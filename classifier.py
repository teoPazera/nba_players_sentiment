import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def _build_alias_patterns(player_aliases, short_alias_blocklist=None):
    """Compile regex patterns for alias-based player mention detection."""
    if short_alias_blocklist is None:
        short_alias_blocklist = set()

    patterns = []
    seen = set()

    for canonical, aliases in player_aliases.items():
        aliases = aliases or []
        all_names = [canonical] + list(aliases)

        for alias in all_names:
            if not alias or not alias.strip():
                continue

            alias_clean = alias.strip()
            alias_lower = alias_clean.lower()

            if alias_lower in short_alias_blocklist:
                continue

            key = (canonical, alias_lower)
            if key in seen:
                continue
            seen.add(key)

            # allow flexible whitespace for multi-word aliases
            escaped = re.escape(alias_clean).replace(r"\ ", r"\s+")
            pattern = r"(?<!\w)" + escaped + r"(?!\w)"

            patterns.append({
                "canonical": canonical,
                "alias": alias_clean,
                "regex": re.compile(pattern, flags=re.IGNORECASE),
            })

    return patterns

def detect_player_mentions(text, alias_patterns):
    """Return a sorted list of canonical player names detected in `text`."""
    found = set()
    for p in alias_patterns:
        if p["regex"].search(text):
            found.add(p["canonical"])
    return sorted(found)


def choose_target(primary_player, mentions, resolved_player=None, fallback_to_primary=True):
    """Pick a target for ABSA scoring (resolved > explicit primary > single mention > fallback)."""
    if resolved_player:
        return resolved_player, "resolved_player"

    if primary_player and primary_player in mentions:
        return primary_player, "explicit_primary_mention"

    if len(mentions) == 1:
        # Sentence names exactly one player, but it might not be your primary.
        return mentions[0], "single_mention_in_sentence"

    if fallback_to_primary and primary_player:
        return primary_player, "fallback_primary"

    return None, "no_target"


def _pipeline_output_to_scores(raw):
    """Normalize HF pipeline output into {'Negative','Neutral','Positive'} score dict."""
    if isinstance(raw, dict):
        # Some custom pipelines/model wrappers may return a dict already.
        scores = dict(raw)
        return {
            "Negative": float(scores.get("Negative", scores.get("NEGATIVE", 0.0))),
            "Neutral": float(scores.get("Neutral", scores.get("NEUTRAL", 0.0))),
            "Positive": float(scores.get("Positive", scores.get("POSITIVE", 0.0))),
        }

    # HF pipeline usually returns list[dict] or list[list[dict]] with top_k=None
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]

    scores = {}
    if isinstance(raw, list):
        for d in raw:
            if isinstance(d, dict) and "label" in d and "score" in d:
                scores[d["label"]] = float(d["score"])

    return {
        "Negative": float(scores.get("Negative", scores.get("NEGATIVE", 0.0))),
        "Neutral": float(scores.get("Neutral", scores.get("NEUTRAL", 0.0))),
        "Positive": float(scores.get("Positive", scores.get("POSITIVE", 0.0))),
    }


def scores_to_value_and_label(scores):
    """Convert class scores to a signed value and argmax label."""
    neg = scores.get("Negative", 0.0)
    neu = scores.get("Neutral", 0.0)
    pos = scores.get("Positive", 0.0)

    value = pos - neg  # expected value with mapping Negative=-1, Neutral=0, Positive=+1

    label = max([("Negative", neg), ("Neutral", neu), ("Positive", pos)], key=lambda x: x[1])[0]
    return value, label


def vader_score_sentence(sentence, pos_th=0.05, neg_th=-0.05):
    """Return VADER sentiment for a sentence (compound score + label)."""
    analyzer = _get_vader_analyzer()
    s = analyzer.polarity_scores(sentence)

    compound = float(s["compound"])
    if compound >= pos_th:
        label = "Positive"
    elif compound <= neg_th:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "model": "vader",
        "scores": {
            "neg": float(s["neg"]),
            "neu": float(s["neu"]),
            "pos": float(s["pos"]),
            "compound": compound,
        },
        "value": compound,
        "label": label,
    }


_VADER_ANALYZER = None


def _get_vader_analyzer() -> SentimentIntensityAnalyzer:
    global _VADER_ANALYZER
    if _VADER_ANALYZER is None:
        _VADER_ANALYZER = SentimentIntensityAnalyzer()
    return _VADER_ANALYZER


class DebertaABSAScorer:
    """Target-aware ABSA scorer using a DeBERTa sequence classification model."""

    def __init__(
        self,
        player_aliases,
        model_id="yangheng/deberta-v3-base-absa-v1.1",
        device=None,
        *,
        local_files_only: bool = False,
    ):
        self.alias_patterns = _build_alias_patterns(player_aliases)

        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        self.model_id = model_id
        self.device = device

        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, local_files_only=local_files_only)

        self.classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

    def score_sentence(self, sentence, primary_player, resolved_player=None, fallback_to_primary=True):
        mentions = detect_player_mentions(sentence, self.alias_patterns)
        target, target_source = choose_target(
            primary_player=primary_player,
            mentions=mentions,
            resolved_player=resolved_player,
            fallback_to_primary=fallback_to_primary,
        )

        if not target:
            return {
                "model": self.model_id,
                "target": None,
                "target_source": target_source,
                "mentions": mentions,
                "scores": None,
                "value": None,
                "label": None,
            }

        # Get all label probabilities (Negative/Neutral/Positive)
        raw = self.classifier(sentence, text_pair=target, top_k=None)
        scores = _pipeline_output_to_scores(raw)
        value, label = scores_to_value_and_label(scores)

        other_mentions = [m for m in mentions if m != target]

        return {
            "model": self.model_id,
            "target": target,
            "target_source": target_source,
            "primary_player": primary_player,
            "mentions": mentions,
            "other_mentions": other_mentions,
            "scores": scores,     # {"Negative": .., "Neutral": .., "Positive": ..}
            "value": value,       # in [-1,+1] approximately
            "label": label,       # argmax class
        }


# --------- Example usage ----------
if __name__ == "__main__":
    player_aliases = {
        "LeBron James": ["lebron", "bron", "king james"],
        "Anthony Davis": ["anthony davis", "ad", "a.d."],
    }

    sentence = "AD is a monster on defense but he disappears in the 4th."

    print(vader_score_sentence(sentence))
    print(DebertaABSAScorer(player_aliases).score_sentence(sentence, primary_player="Anthony Davis"))
