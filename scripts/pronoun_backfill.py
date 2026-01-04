"""Heuristic pronoun backfill utilities (not full coreference resolution)."""

from typing import Dict, List, Optional, Set

from article_player_split import normalize


DEFAULT_PRONOUNS: Set[str] = {"he", "him", "his", "himself"}


def sentence_has_pronoun(sent, pronouns: Optional[Set[str]] = None) -> bool:
    pronouns = DEFAULT_PRONOUNS if pronouns is None else pronouns
    for tok in sent:
        if tok.lower_ in pronouns:
            return True
    return False


def other_persons_in_sentence(sent, alias_to_player: Dict[str, str]) -> List[str]:
    other_persons: List[str] = []
    for ent in sent.ents:
        if ent.label_ != "PERSON":
            continue
        norm = normalize(ent.text)
        if norm in alias_to_player:
            continue
        if ent.text not in other_persons:
            other_persons.append(ent.text)
    return other_persons


def build_pronoun_backfill_record(
    sent,
    alias_to_player: Dict[str, str],
    context_player: str,
    *,
    other_persons: Optional[List[str]] = None,
    tie_breaker: str = "pronoun_backfill",
) -> Optional[Dict[str, object]]:
    sent_text = sent.text.strip()
    if not sent_text:
        return None

    if other_persons is None:
        other_persons = other_persons_in_sentence(sent, alias_to_player)

    return {
        "sentence": sent_text,
        "players": [context_player],
        "primary_player": context_player,
        "tie_breaker": tie_breaker,
        "other_persons": other_persons,
    }
