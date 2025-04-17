from analyzer.twitter_client import fetch_tweets

try:
    tweets = fetch_tweets("السعودية", 1)
    print("Success! First tweet:", tweets[0].full_text)
except Exception as e:
    print("Error:", str(e))