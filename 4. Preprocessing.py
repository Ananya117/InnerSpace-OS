import pandas as pd
import re

# Load dataset
df = pd.read_csv("emotions_cleaned.csv")
df = df[['text', 'mood_label']].dropna()

# Convert emotion to binary mood
def convert_to_mood(emotion):
    if emotion in ["joy", "surprise"]:
        return "Good"
    else:
        return "Not-Good"

df['mood_label'] = df['mood_label'].apply(convert_to_mood)

# Clean text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # remove links
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text.lower()

df['text'] = df['text'].apply(clean_text)

# Optional: Remove too short/long entries
df = df[df['text'].str.split().apply(len).between(3, 70)]

# Save cleaned data
df.to_csv("emotions_preprocessed_final.csv", index=False)
