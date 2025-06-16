import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load model + tokenizer from the extracted folder
MODEL_PATH = "binary_mood_model"

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model()

# Set Streamlit page config
st.set_page_config(page_title="Innerspace Mood Detector", page_icon="🧠", layout="centered")

# UI
st.title("🧘 Innerspace Mood Classifier")
st.markdown("Enter a journal entry below and find out if your mood is detected as **Good** or **Not-Good**.")

# Input
user_input = st.text_area("📝 Your Journal Entry", height=150, placeholder="Write freely...")

if st.button("Analyze Mood"):
    if not user_input.strip():
        st.warning("Please enter something.")
    else:
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        # Display result
        if pred == 1:
            st.success(f"🌈 Mood: **Good** — Confidence: `{confidence:.2%}`")
        else:
            st.error(f"🌧️ Mood: **Not-Good** — Confidence: `{confidence:.2%}`")

        # Optional: Show raw probabilities
        with st.expander("See Prediction Details"):
            st.write(f"Good (1): `{probs[0][1].item():.4f}`")
            st.write(f"Not-Good (0): `{probs[0][0].item():.4f}`")
