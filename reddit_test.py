import os

import praw
from dotenv import load_dotenv
from prawcore.exceptions import ResponseException

load_dotenv()

client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT", "nba-sentiment-research")

if not client_id or not client_secret:
    raise SystemExit(
        "Missing Reddit credentials. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET (e.g., in a local .env)."
    )

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent,
    check_for_async=False,  # avoids a warning in some environments
)

print("read_only:", reddit.read_only)

try:
    sub = reddit.subreddit("nba")
    # These lines force API calls
    print("display_name:", sub.display_name)
    print("title:", sub.title)
    print("OK - credentials work.")
except ResponseException as e:
    print("Got ResponseException:", e)
