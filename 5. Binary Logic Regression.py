import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load preprocessed binary data
df = pd.read_csv("emotions_preprocessed_final.csv")
df['label'] = df['mood_label'].apply(lambda x: 1 if x == "Good" else 0)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_vec, y_train)

# Predict & Evaluate
y_pred = lr.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Not-Good", "Good"]))
