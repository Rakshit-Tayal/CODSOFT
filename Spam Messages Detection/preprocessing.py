import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()  # No word_tokenize here
    filtered = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(filtered)
