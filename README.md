# Fake News Detection using NLP

This project is a **Fake News Detection system** built using **Basic Natural Language Processing (NLP)** and **Machine Learning** techniques.  
It classifies a news article as **FAKE** or **REAL** based on its textual content.

The goal of this project is to demonstrate **text preprocessing, feature extraction, and classification** on a real-world dataset.

---

## Project Overview

Fake news has become a serious issue in today’s digital world.  
This project helps identify misleading or false news articles using:

- Text cleaning and preprocessing
- TF-IDF vectorization
- Logistic Regression classification

The model achieves **~95% accuracy**, making it both reliable and interview-ready.

---

## Project Structure
Fake news detection using NLP/
- data.csv
- train.py
- predict.py
- requirements.txt
- model/
    - fake_news_model.pkl
    - tfidf_vectorizer.pkl

---

## Dataset

- Source: Fake & Real News Dataset (Kaggle)
- Columns used:
  - `text` – news article content
  - `subject` – category of the news

Since the dataset does not contain an explicit label column, labels were **derived from the subject field**:
- Certain subjects were mapped as **FAKE**
- Remaining subjects were mapped as **REAL**

This reflects a real-world scenario where labels must be engineered.

---

## Methodology

### 1. Text Preprocessing
- Converted text to lowercase
- Removed punctuation and numbers
- Removed extra whitespaces

### 2. Feature Extraction
- Used **TF-IDF Vectorizer**
- Limited to top 5000 features
- English stopwords removed internally

### 3. Model Training
- Algorithm: **Logistic Regression**
- Train–test split: 80% / 20%

### 4. Evaluation
- Accuracy: ~95%
- Metrics: Precision, Recall, F1-Score

---

## Installation

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```
--- 

## How to Run
### 1.Train the Model
```bash
python train.py
```

This will:
- Train the model
- Evaluate performance
- Save the trained model and vectorizer in the model/ folder

### 2.Predict Fake or Real News
```bash
python predict.py
```

Paste a news article when prompted:
```bash
Enter news article text:
```
Output:
```bash
Prediction: FAKE
or
Prediction: REAL
``` 
---

#### Sample Results
```bash
Accuracy: 0.955
FAKE: Precision 0.95 | Recall 0.98
REAL: Precision 0.97 | Recall 0.91
``` 

Example Input
A secret government document has revealed that all citizens will be required to install tracking chips in their phones by next month.

Prediction: FAKE

---

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Joblib

## Key Learnings
- Handling real-world datasets without explicit labels
- Text preprocessing for NLP tasks
- Feature extraction using TF-IDF
- Building and evaluating ML classification models
- Saving and loading trained models for inference

## Future Improvements
- Add Naive Bayes / SVM models
- Implement lemmatization
- Add confidence score for predictions
- Build a Streamlit web application
- Extend to deep learning models (LSTM / BERT)
