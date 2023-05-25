import string
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Split text into token with preprocessing
nltk.download("stopwords")
nltk.download("vader_lexicon")
text = open("text_data.txt", encoding="utf-8").read()
text = text.lower()
text = text.translate(str.maketrans("", "", string.punctuation))
text_token = word_tokenize(text)
text_token = [token for token in text_token if token not in stopwords.words("english")]

emotion_list = []
with open("emotions.txt", "r") as file:
    for line in file:
        line = line.replace("\n", "").replace(",", "").replace("'", "").strip()
        word, emotion = line.split(":")
        if word in text_token:
            emotion_list.append(emotion)


def sentiment_analyze(text):
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    neg = score["neg"]
    pos = score["pos"]
    if neg > pos:
        print("Negative Sentiment")
    elif pos > neg:
        print("Positive Sentiment")
    else:
        print("Neutral Sentiment")


sentiment_analyze(text)
count = Counter(emotion_list)
plt.bar(count.keys(), count.values())
plt.margins(x=0)
plt.savefig("bar.png")
