# Fake News Detection using Machine Learning

## Overview
This project aims to detect **fake news articles** using machine learning techniques. It utilizes **TF-IDF vectorization** to extract features from text data and trains a **Passive Aggressive Classifier** to classify news as real or fake.

## Dataset
- The dataset is uploaded as a ZIP file (`news.zip`) in **Google Colab**.
- It is extracted and loaded into a **Pandas DataFrame**.
- The dataset should contain at least two columns:
  - `text`: The news content
  - `label`: The classification (**FAKE** or **REAL**)

## Dependencies
Install the required libraries using:
```bash
pip install scikit-learn pandas numpy
```

## Steps to Run the Project
1. **Upload the Dataset**: Upload `news.zip` to **Google Colab**.
2. **Extract & Load Data**: The script extracts the ZIP file and loads the dataset into a DataFrame.
3. **Preprocess Data**: The text is vectorized using **TF-IDF**.
4. **Train the Model**: A **Passive Aggressive Classifier** is trained on the dataset.
5. **Evaluate Performance**: The model predicts labels for the test data, and accuracy is computed.

## Code Snippet
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.25, random_state=7)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
```

## Results
- The model provides an accuracy score based on the given dataset.
- A **confusion matrix** is displayed to evaluate classification performance.

## Future Improvements
- Use **Deep Learning (LSTMs, Transformers)** for better text classification.
- Include **more features** like metadata (author, source credibility, etc.).
- Implement **real-time detection** for news articles.

