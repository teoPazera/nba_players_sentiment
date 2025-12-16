import time
import json
from datetime import datetime, timezone

import pandas as pd
import praw
import spacy
from spacy.matcher import PhraseMatcher
import re
from collections import defaultdict


# ====== CONFIG ======

MAX_PEOPLE_PER_POST = 2
MAX_PEOPLE_PER_COMMENT = 2
MAX_POSTS_PER_PLAYER_SUB = 50   # cap per (player, subreddit) during scraping
MAX_COMMENTS_PER_POST = 30

TOP_POSTS_PER_PLAYER = 50   

ALIAS_CSV = r"data\player_data\player_aliases.csv"
SIGNINGS_CSV = r"data\player_data\fox_free_agency_signings_2025.csv"
# OUT_JSON = f"data\reddit_data\reddit_player_posts_may_sep_2025_with_comments_top.json"
OUT_JSON = f"data/reddit_data/reddit_player_posts_may_sep_2025_{MAX_POSTS_PER_PLAYER_SUB}-{TOP_POSTS_PER_PLAYER}-{MAX_PEOPLE_PER_POST}-{MAX_COMMENTS_PER_POST}-{MAX_PEOPLE_PER_COMMENT}.json"
# name posts_per 
SUBREDDITS = ["nba", "nbadiscussion", "NBAtalk"]

GLOBAL_START_DT = datetime(2025, 5, 1, tzinfo=timezone.utc)
     # adjustable


# ====== helper funcs ======

def normalize(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def find_person_entities(text, nlp):
    """
    Return a list of PERSON entity strings as detected by spaCy NER.
    """
    doc = nlp(text)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    # optional: normalize & dedupe
    norm_seen = set()
    unique_persons = []
    for p in persons:
        norm = normalize(p)
        if norm and norm not in norm_seen:
            norm_seen.add(norm)
            unique_persons.append(p)
    return unique_persons



def load_alias_maps(path=ALIAS_CSV):
    """
    Returns:
      - alias_to_player: normalized alias -> canonical player name (for matching in text)
      - player_to_aliases: canonical player name -> set of original alias strings (for building search queries)
    """
    df = pd.read_csv(path)

    alias_to_player = {}
    player_to_aliases = {}

    for _, row in df.iterrows():
        alias_raw = str(row["alias"])
        player = str(row["player"])

        alias_norm = normalize(alias_raw)
        if alias_norm:
            alias_to_player[alias_norm] = player

        player_to_aliases.setdefault(player, set()).add(alias_raw)

    # make sure full name is included too
    for player in player_to_aliases.keys():
        player_to_aliases[player].add(player)

    return alias_to_player, player_to_aliases


def create_player_matcher(nlp, alias_to_player):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = []
    for alias_norm in alias_to_player.keys():
        if len(alias_norm) <= 2:
            continue  # skip super-short aliases
        patterns.append(nlp.make_doc(alias_norm))
    matcher.add("PLAYER", patterns)
    return matcher


def find_players_in_text_spacy(text, nlp, matcher, alias_to_player):
    doc = nlp(text)
    found_players = []
    for match_id, start, end in matcher(doc):
        span = doc[start:end]
        alias_norm = normalize(span.text)
        player = alias_to_player.get(alias_norm)
        if player:
            found_players.append(player)
    # dedupe but preserve order
    return list(dict.fromkeys(found_players))


def load_signing_dates(path=SIGNINGS_CSV):
    """
    Load signing dates and map them by normalized player name.
    Returns: dict norm_player -> signing_timestamp (UTC)
    """
    df = pd.read_csv(path)

    # require signing_date
    df = df.dropna(subset=["signing_date"])
    df["signing_date_dt"] = pd.to_datetime(df["signing_date"], errors="coerce")
    df = df.dropna(subset=["signing_date_dt"])

    signing_map = {}
    for _, row in df.iterrows():
        player = str(row["player"])
        norm_player = normalize(player)
        dt = row["signing_date_dt"]
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        signing_map[norm_player] = dt.timestamp()

    print(f"Loaded signing dates for {len(signing_map)} players from {path}")
    return signing_map


# ====== load alias map, signing dates, NER ======

alias_to_player, player_to_aliases = load_alias_maps(ALIAS_CSV)
print(f"Loaded {len(alias_to_player)} aliases for {len(player_to_aliases)} players.")

signing_ts_map = load_signing_dates(SIGNINGS_CSV)

nlp = spacy.load("en_core_web_sm")
matcher = create_player_matcher(nlp, alias_to_player)


# ====== Reddit auth (userless) ======

reddit = praw.Reddit(
    client_id="xN75Y0EZ8Uo2SG8gcUeW3g",
    client_secret="i9j6f-hoeCEWxBdLyXxdIuBBDSAeqg",
    user_agent="nba research diag script by u/Hadak69",
    check_for_async=False,
)

print("Read-only:", reddit.read_only)


# ====== main scrape ======

posts_by_player = defaultdict(list)

for player, aliases in player_to_aliases.items():
    norm_player = normalize(player)
    signing_ts = signing_ts_map.get(norm_player)

    # only search for players with signing date
    if signing_ts is None:
        continue

    player_start_ts = GLOBAL_START_DT.timestamp()
    player_end_ts = signing_ts
    if player_end_ts <= player_start_ts:
        continue

    # build search query: OR of all aliases
    terms = []
    for alias in aliases:
        alias = alias.strip()
        if not alias:
            continue
        if " " in alias:
            terms.append(f'"{alias}"')
        else:
            terms.append(alias)

    if not terms:
        continue

    query = " OR ".join(terms)
    print(
        f"\n=== Searching for {player} "
        f"(window: {datetime.fromtimestamp(player_start_ts)} to {datetime.fromtimestamp(player_end_ts)})"
    )
    print(f"    query: {query}")

    for sub_name in SUBREDDITS:
        subreddit = reddit.subreddit(sub_name)
        print(f"  in r/{sub_name}...")

        count_for_this_combo = 0

        for submission in subreddit.search(
            query=query,
            sort="new",
            time_filter="year",
            limit=None,
        ):
            created = submission.created_utc

            # per-player post time window filter
            if created < player_start_ts or created > player_end_ts:
                continue

            title = submission.title or ""
            body = submission.selftext or ""
            combined_text = f"{title}\n\n{body}"

            # 1) tracked players (your existing logic)
            matched_players = find_players_in_text_spacy(
                combined_text, nlp, matcher, alias_to_player
            )
            if not matched_players:
                continue
            if player not in matched_players:
                continue

            num_players = len(matched_players)
            
            post_type = "single" if num_players == 1 else "multi"

            # 2) independent PERSON entities
            all_persons = find_person_entities(combined_text, nlp)
            num_persons_total 
            num_persons_total = len(all_persons)

            # we approximate: tracked players ≈ subset of those PERSON entities
            num_other_persons = max(0, num_persons_total - num_players)
            if num_persons_total > MAX_PEOPLE_PER_POST:
                continue
            
            # ====== fetch comments, also filtered by signing window ======
            comments_data = []
            try:
                submission.comments.replace_more(limit=0)

                bucket_search_player = []
                bucket_zero = []

                for comment in submission.comments.list():
                    c_created = comment.created_utc
                    if c_created < player_start_ts or c_created > player_end_ts:
                        continue

                    c_body = comment.body or ""
                    c_players = find_players_in_text_spacy(
                        c_body, nlp, matcher, alias_to_player
                    )
                    c_num_players = len(c_players)
                    

                    c_persons = find_person_entities(c_body, nlp)
                    c_num_persons_total = len(c_persons)
                    c_num_other_persons = max(0, c_num_persons_total - c_num_players)

                    if c_num_persons_total > MAX_PLAYERS_PER_COMMENT:
                        continue

                    record = {

                        "comment_id": comment.id,
                        "parent_id": comment.parent_id,
                        "created_utc": c_created,
                        "created_date_utc": datetime.utcfromtimestamp(c_created).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "body": c_body,
                        "matched_players": c_players,
                        "num_matched_players": c_num_players,
                        "c_num_other_persons": c_num_other_persons
                    }

                    if player in c_players:
                        bucket_search_player.append(record)
                    elif c_num_players == 0 and c_num_persons_total == 0:
                        bucket_zero.append(record)
                    else:
                        # comment talks about other player(s) but not our player
                        # you can ignore or store separately
                        pass

                # sort buckets (example: by score descending, then newest first)
                def sort_key(rec):
                    # negative for descending score/time
                    score = rec["c_num_other_persons"] 
                    return (-score, -rec["created_utc"])

                bucket_search_player.sort(key=sort_key)
                bucket_zero.sort(key=sort_key)

                selected = []

                for rec in bucket_search_player:
                    if len(selected) >= MAX_COMMENTS_PER_POST:
                        break
                    selected.append(rec)

                if len(selected) < MAX_COMMENTS_PER_POST:
                    needed = MAX_COMMENTS_PER_POST - len(selected)
                    selected.extend(bucket_zero[:needed])

                comments_data = selected

            except Exception as e:
                print(f"    error fetching comments for {submission.id}: {e}")
                comments_data = []

            post_record = {
                "search_player": player,
                "subreddit": sub_name,
                "post_id": submission.id,
                "post_url": f"https://www.reddit.com{submission.permalink}",
                "created_utc": created,
                "created_date_utc": datetime.utcfromtimestamp(created).strftime("%Y-%m-%d %H:%M:%S"),
                "title": title,
                "selftext": body,
                "matched_players": matched_players,
                "num_matched_players": num_players,
                "post_type": post_type,
                "all_persons": all_persons,
                "num_persons_total": num_persons_total,
                "num_other_persons": num_other_persons,
                "comments": comments_data,
            }


            posts_by_player[player].append(post_record)

            count_for_this_combo += 1
            print(
                f"{datetime.utcfromtimestamp(created).strftime('%Y-%m-%d')} "
                f"[{sub_name}] {title[:80]}... -> {matched_players} "
                f"(type: {post_type}, comments: {len(comments_data)})"
            )

            if count_for_this_combo >= MAX_POSTS_PER_PLAYER_SUB:
                print(
                    f"    reached {MAX_POSTS_PER_PLAYER_SUB} posts for {player} in r/{sub_name}, moving on."
                )
                break

            time.sleep(0.2)


# ====== rank & keep top N posts per player ======

final_posts = []

for player, plist in posts_by_player.items():
    if not plist:
        continue

    # sort: fewer players mentioned first, then newer posts first
    plist_sorted = sorted(plist, key=lambda p: (
        p.get("num_other_persons", 0),       # fewer other persons
        p["num_matched_players"],            # fewer tracked players (ideally 1)
        -p["created_utc"],                   # newer posts as tiebreaker
    ),
)



    top_n = plist_sorted[:TOP_POSTS_PER_PLAYER]
    final_posts.extend(top_n)

print(f"\nTotal posts collected before filtering: {sum(len(v) for v in posts_by_player.values())}")
print(f"Total posts kept after top-{TOP_POSTS_PER_PLAYER} per player: {len(final_posts)}")


# ====== save as JSON ======

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(final_posts, f, ensure_ascii=False, indent=2)

print(f"Saved {len(final_posts)} posts (with comments) to {OUT_JSON}")
