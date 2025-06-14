#Logistic Regression for Mood Classification
# This script trains a logistic regression model to classify text into mood categories.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load and clean data
df = pd.read_csv("Data/emotions_cleaned.csv")
df = df[['text', 'mood_label']].dropna()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['mood_label'], test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))
