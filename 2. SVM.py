#2.Support Vector Machine (TF-IDF)

from sklearn.svm import SVC

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['mood_label'], test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train SVM
svm_model = SVC()
svm_model.fit(X_train_vec, y_train)

# Evaluate
y_pred_svm = svm_model.predict(X_test_vec)
print(classification_report(y_test, y_pred_svm))