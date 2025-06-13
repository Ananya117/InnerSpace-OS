### **1. Logistic Regression**

**→ Traditional Machine Learning Baseline**

* **What Was Done:**

  * Used `TfidfVectorizer` to convert journal-like sentences to feature vectors.
  * Trained a Logistic Regression classifier on the cleaned emotion dataset.

* **Dataset Used:**

  * Custom cleaned CSV dataset (`emotions_cleaned.csv`)
  * Format: `"text"` and `"mood_label"` columns
  * \~6 emotion labels (joy, sadness, anger, etc.)

* **Problem Faced:**

  * Dataset was highly **imbalanced** → majority predicted as "joy"
  * Accuracy plateaued around **74%**
  * Couldn’t handle subtle emotional tones in full sentences

---

### **2. Support Vector Machine (SVM)**

**→ Alternative Classical ML Approach**

* **What Was Done:**

  * Retried the same dataset and TF-IDF pipeline using an SVM model
  * Evaluated performance to test non-linear classification ability

* **Dataset Used:**

  * Same as above (`emotions_cleaned.csv`)

* **Problem Faced:**

  * Accuracy **dropped to \~72%**
  * Still overfitting on dominant classes
  * No major improvement over Logistic Regression
  * Couldn’t scale well for more complex journaling sentences

---

### **3. DistilBERT (Transformer Fine-Tuning)**

**→ Final Production-Ready Deep Learning Model**

* **What Was Done:**

  * Loaded `distilbert-base-uncased` using Hugging Face
  * Filtered **GoEmotions** dataset to keep:

    * Only **6 emotions**: `anger, disgust, fear, joy, sadness, surprise`
    * Only **single-label entries**
  * Used `Trainer` API to fine-tune the model for sequence classification
  * Tokenized using `AutoTokenizer`, padded using `DataCollatorWithPadding`

* **Dataset Used:**

  * [GoEmotions – Hugging Face](https://huggingface.co/datasets/go_emotions)
  * Filtered version used for training (\~3.4k samples after cleaning)

* **Problem Faced:**

  * Model still leaned towards **joy/sadness** in journal-style writing
  * Classifier relied heavily on **keywords/punctuation**, not full sentence intent
  * Long input sentences confused some predictions (esp. disgust/fear)
  * Needs simplification for user-friendly feedback

