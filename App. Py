Fake News Detection using NLP and Machine Learning
# Dataset Source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

# Install dependencies
# !pip install pandas numpy scikit-learn nltk gradio

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
import gradio as gr
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
# Download and unzip the dataset from https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
real_news = pd.read_csv("True.csv")
fake_news = pd.read_csv("Fake.csv")

# Add labels
real_news["label"] = 0
fake_news["label"] = 1

# Combine datasets
df = pd.concat([real_news, fake_news], axis=0).reset_index(drop=True)
df = df[['text', 'label']].dropna()

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['cleaned_text'] = df['text'].apply(preprocess)

# Vectorization
X = df['cleaned_text']
y = df['label']

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Gradio Interface
def detect_fake_news(text):
    processed = preprocess(text)
    vec = tfidf.transform([processed])
    prediction = model.predict(vec)[0]
    return "Fake News" if prediction == 1 else "Real News"

interface = gr.Interface(
    fn=detect_fake_news,
    inputs=gr.Textbox(lines=5, placeholder="Paste news article or headline here..."),
    outputs="text",
    title="Fake News Detector",
    description="Detect whether a news article is fake or real using NLP and a trained Logistic Regression model."
)

interface.launch()
