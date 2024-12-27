import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import pandas as pd

# Download required datasets (if not already downloaded)
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Preprocessing function
def preprocess_text(text):
    if pd.isna(text):  # Handle missing values
        return ""

    # 1. Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 3. Convert to lowercase
    text = text.lower()

    # 4. Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)

    # 5. Tokenize using split (replacing word_tokenize)
    tokens = text.split()

    # 6. Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # 7. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # 8. Rejoin tokens to form the cleaned text
    cleaned_text = ' '.join(tokens)

    return cleaned_text
