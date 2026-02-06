from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(self, text: str) -> float:
        """
        Analyzes the sentiment of a given text using VADER.
        Returns a compound score between -1 (Negative) and 1 (Positive).
        """
        if not text:
            return 0.0
        
        scores = self.analyzer.polarity_scores(text)
        return scores['compound']

# Singleton instance
analyzer = SentimentAnalyzer()
