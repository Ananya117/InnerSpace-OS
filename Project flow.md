### **1. Logistic Regression**

**→ Traditional Machine Learning Baseline**

* **What Was Done:**

  * Used `TfidfVectorizer` to convert journal-like sentences to feature vectors.
  * Trained a Logistic Regression classifier on the cleaned emotion dataset.

* **Dataset Used:**

  * Custom cleaned CSV dataset (`emotions_cleaned.csv`)
  * Format: `"text"` and `"mood_label"` columns
  * \~6 emotion labels (joy, sadness, anger, etc.)

* **Accuracy:**

  * \~74%

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

* **Accuracy:**

  * \~72%

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

* **Accuracy:**

  * \~84% (on test split from GoEmotions subset)

* **Problem Faced:**

  * Model still leaned towards **joy/sadness** in journal-style writing
  * Classifier relied heavily on **keywords/punctuation**, not full sentence intent
  * Long input sentences confused some predictions (esp. disgust/fear)
  * Needs simplification for user-friendly feedback

---

### **4. Preprocessing (Binary Phase)**

**→ Cleaned & Refined for Good vs Not-Good Classification**

* **What Was Done:**

  * Removed links, emojis, and special characters using regex
  * Converted all text to lowercase
  * Dropped rows with extremely short or excessively long journal entries
  * Mapped original emotion labels to binary classes:

    * *Good* → `joy`, `surprise`
    * *Not-Good* → `anger`, `sadness`, `disgust`, `fear`
  * Saved final version as `emotions_preprocessed_final.csv`

* **Problem Faced:**

  * Some entries were ambiguous or sarcastic
  * Filtering removed a few meaningful longer entries
  * Class imbalance still existed → more Good entries

---

### **5. Logistic Regression (Binary)**

**→ Traditional ML Baseline for Good vs Not-Good Mood**

* **What Was Done:**

  * Used `TfidfVectorizer` to convert journal entries to numerical vectors
  * Trained a Logistic Regression model
  * Evaluated on binary labels (*Good* vs *Not-Good*)
  * Used `train_test_split()` for an 80-20 evaluation

* **Dataset Used:**

  * `emotions_preprocessed_final.csv`
  * Format: `"text"` and binary `"label"` columns
  * \~3.4k entries after full preprocessing

* **Accuracy:**

  * \~77%

* **Problem Faced:**

  * Subtle or poetic mood variations weren’t caught well
  * High bias towards more frequent "Good" class

---

### **6. DistilBERT (Binary Fine-Tuning)**

**→ Transformer-Based Deep Learning Model**

* **What Was Done:**

  * Used `distilbert-base-uncased` from Hugging Face
  * Tokenized and padded using `DistilBertTokenizerFast`
  * Trained using the `Trainer` API with PyTorch backend
  * Binary classification task: *Good* vs *Not-Good*
  * Early stopping not used yet (basic 3-epoch fine-tune)

* **Dataset Used:**

  * `emotions_preprocessed_final.csv`
  * Tokenized and converted to Hugging Face `Dataset` object
  * Class balance not manually adjusted yet

* **Accuracy:**

  * \~90.4% (on validation split using Trainer.predict)

* **Problem Faced:**

  * Needed Colab GPU → training on CPU was too slow
  * Long poetic or emotionally complex entries sometimes got misclassified
  * Still showed keyword dependency instead of deeper context understanding

---

### **7. Checkpoint-Based Evaluation on Colab**

**→ Reuse & Evaluate Trained DistilBERT Model**

* **What Was Done:**

  * Uploaded model checkpoint (`innerspace_model.zip`)
  * Extracted, loaded model and tokenizer with `.from_pretrained()`
  * Re-tokenized validation set and passed it to the model using `Trainer.predict()`
  * Generated `classification_report` and confusion matrix

* **Dataset Used:**

  * `emotions_preprocessed_final.csv`
  * Same validation split reused from training

* **Accuracy:**

  * \~90.4% (verified during checkpoint evaluation)

* **Problem Faced:**

  * Runtime restarts needed after `datasets` library install
  * Folder structure must be preserved inside the zip
  * Still confused by entries with sarcasm or mixed emotions
    
---

### **8. Lightweight DistilBERT (Colab Version for Deployment)**

**→ Clean Binary Model for Streamlit Use**

* **What Was Done:**

  * Final cleaned binary mood classifier using `DistilBertForSequenceClassification`
  * Installed and fixed correct `transformers==4.40.1` to avoid compatibility errors
  * Processed CSV file (`clean_text`, `mood_label`) and mapped emotions to binary
  * Tokenized with `DistilBertTokenizerFast`, trained using Hugging Face `Trainer`
  * Metrics: accuracy, precision, recall, f1-score calculated
  * Exported the trained model and tokenizer to ZIP (`binary_mood_model.zip`) for reuse

* **Dataset Used:**

  * `emotions_preprocessed_final.csv`
  * Columns: `"clean_text"` and `"mood_label"` mapped to binary label (Good vs Not-Good)
  * Slightly imbalanced but not manually adjusted yet

* **Accuracy:**

  * \~76.3%
    (`f1: 0.771`, `precision: 0.747`, `recall: 0.797`)

* **Problem Faced:**

  * Hugging Face version mismatch caused `Trainer` to break initially
  * Installing `transformers` led to repeated runtime restarts
  * Tokenization/label split errors due to session reset and missing variables
  * Accuracy lower than expected (compared to GoEmotions fine-tune), but model still usable
  * **Final model saved + zipped successfully for Streamlit deployment**
 
---

### **9. Local Streamlit Deployment + Modular Journal Integration**

**→ Local Innerspace OS Build Begins**

**What Was Done:**

* Moved the exported `binary_model` folder (from the Colab-trained ZIP) into a local `model/` directory.
* Created a lightweight **`mood_detector/mood.py`** to handle inference using `DistilBertTokenizerFast` and `DistilBertForSequenceClassification`.
* Built an initial **Streamlit UI (`app.py`)** to load the model and test predictions.
* Created modular directory structure:

  * `journal/journal.py` for a mood-aware journaling interface
  * `mood_detector/mood.py` for prediction logic
  * `model/binary_model/` containing tokenizer & model files
* Integrated everything in a central **Streamlit app** with routing to the `journal_panel()` interface.
* Verified local run using:

  ```
  streamlit run app.py
  ```
* Saved all data entries into a CSV (`journal_entries.csv`) with timestamps and predicted moods.

**Folder Structure:**

```
Innerspace/
├── app.py
├── model/
│   └── binary_model/
├── mood_detector/
│   └── mood.py
├── journal/
│   └── journal.py
```

**Problem Faced:**

* `set_page_config()` placement caused `StreamlitSetPageConfigMustBeFirstCommandError`—resolved by ensuring it's the first call.
* Relative import issues (`ModuleNotFoundError: No module named 'mood_detector'`) fixed using proper `sys.path` and module structure.
* Model loading path needed to match local structure (`model/binary_model`).

**Status:**
Streamlit version now runs **perfectly** on local machine with a modular journal and live mood prediction support.


