from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

MODEL_PATH = "model/binary_model"

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model()

def predict_mood(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Good" if prediction == 1 else "Not-Good"
