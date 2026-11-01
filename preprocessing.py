import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()
# Keep negation words because they can reverse sentiment.
_stop_words = set(ENGLISH_STOP_WORDS) - {"no", "not", "nor", "never"}

_CONTRACTIONS = {
    "didn't": "did not",
    "don't": "do not",
    "doesn't": "does not",
    "isn't": "is not",
    "wasn't": "was not",
    "weren't": "were not",
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "aren't": "are not",
    "it's": "it is",
    "that's": "that is",
    "they're": "they are",
    "we're": "we are",
    "you're": "you are",
    "i'm": "i am",
}

def clean_text(text):
    """Clean a review consistently during training and Streamlit prediction."""
    text = str(text).lower()
    for contraction, expanded in _CONTRACTIONS.items():
        text = text.replace(contraction, expanded)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = re.sub(r"\s+", " ", text).strip().split()
    words = [_stemmer.stem(word) for word in words if word not in _stop_words]
    return " ".join(words)
