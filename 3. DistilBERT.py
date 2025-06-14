#3.DistilBERT (Transformer Fine-Tuning)

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import numpy as np

# Load and filter GoEmotions
dataset = load_dataset("go_emotions", split="train")
target_emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
label_map = dataset.features['labels'].feature.names
label2id = {emotion: idx for idx, emotion in enumerate(target_emotions)}
id2label = {idx: emotion for emotion, idx in label2id.items()}

def filter_labels(example):
    return len(example['labels']) == 1 and label_map[example['labels'][0]] in target_emotions
filtered_dataset = dataset.filter(filter_labels)

def encode_labels(example):
    label_str = label_map[example['labels'][0]]
    example['label'] = label2id[label_str]
    return example
filtered_dataset = filtered_dataset.map(encode_labels)

# Convert and split
df = filtered_dataset.to_pandas()
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize(example):
    return tokenizer(example['text'], truncation=True)
tokenized_train = train_dataset.map(tokenize, batched=True)
tokenized_test = test_dataset.map(tokenize, batched=True)

# Collator and model
collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=6,
    id2label=id2label,
    label2id=label2id
)

# Training arguments
args = TrainingArguments(
    output_dir="bert-output",
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01
)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": (preds == labels).mean()}

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics
)

# Train and evaluate
trainer.train()
preds_output = trainer.predict(tokenized_test)
y_pred = np.argmax(preds_output.predictions, axis=-1)
y_true = preds_output.label_ids
print(classification_report(y_true, y_pred, target_names=target_emotions))

# Save model
model.save_pretrained("model/bert_emotion")
tokenizer.save_pretrained("model/bert_emotion")