import joblib
import re

model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("\n--- Fake News Detection ---")
news_text = input("Enter news article text:\n\n")

cleaned_text = clean_text(news_text)
vectorized_text = vectorizer.transform([cleaned_text])

prediction = model.predict(vectorized_text)

print("\nPrediction:", prediction[0])
