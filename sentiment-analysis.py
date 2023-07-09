import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


nltk.download('vader_lexicon')


def analyze_sentiment(text):

    sia = SentimentIntensityAnalyzer()

    # Perform sentiment analysis
    sentiment_scores = sia.polarity_scores(text)


    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        sentiment = 'Positive'
    elif compound_score <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    # Return the sentiment and scores
    return sentiment, sentiment_scores


# Test the sentiment analysis function
text = "I love this movie. It's fantastic!"
sentiment, scores = analyze_sentiment(text)
print(f"Text: {text}")
print(f"Sentiment: {sentiment}")
print(f"Sentiment Scores: {scores}")