import twitter
from django.conf import settings

def get_twitter_client():
    return twitter.Api(
        consumer_key=settings.TWITTER_API_KEY,
        consumer_secret=settings.TWITTER_API_SECRET,
        access_token_key=settings.TWITTER_ACCESS_TOKEN,
        access_token_secret=settings.TWITTER_ACCESS_SECRET,
        tweet_mode='extended',
        sleep_on_rate_limit=True
    )

def fetch_tweets(hashtag, count=100):
    client = get_twitter_client()
    return client.GetSearch(
        term=f"#{hashtag} -filter:retweets",
        count=count,
        result_type="recent",
        lang="ar"
    )