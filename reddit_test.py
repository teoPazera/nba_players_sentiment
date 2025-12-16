import praw
from prawcore.exceptions import ResponseException

reddit = praw.Reddit(
    client_id='xN75Y0EZ8Uo2SG8gcUeW3g',
    client_secret='i9j6f-hoeCEWxBdLyXxdIuBBDSAeqg',
    user_agent="nba research diag script by u/Hadak69",
    check_for_async=False,  # avoids a warning in some environments
)

print("read_only:", reddit.read_only)

try:
    sub = reddit.subreddit("nba")
    # These lines force API calls
    print("display_name:", sub.display_name)
    print("title:", sub.title)
    print("OK – credentials work.")
except ResponseException as e:
    print("Got ResponseException:", e)
