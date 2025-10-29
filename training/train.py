from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
import evaluate

# ======================
# Hàm đọc file .conll
# ======================
def read_conll_file(filepath):
    sentences = []
    tokens = []
    ner_tags = []

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": ner_tags})
                    tokens, ner_tags = [], []
            else:
                parts = line.split()
                if len(parts) == 2:
                    token, tag = parts
                    tokens.append(token)
                    ner_tags.append(tag)
        if tokens:
            sentences.append({"tokens": tokens, "ner_tags": ner_tags})

    return Dataset.from_dict({
        "tokens": [s["tokens"] for s in sentences],
        "ner_tags": [s["ner_tags"] for s in sentences]
    })


# ======================
# Tạo DatasetDict
# ======================
dataset = DatasetDict({
    "train": read_conll_file("dataset/word/train_word.conll"),
    "validation": read_conll_file("dataset/word/dev_word.conll"),
    "test": read_conll_file("dataset/word/test_word.conll"),
})


model_name = "NlpHUST/ner-vietnamese-electra-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

label_list = [
    "O",
    "B-NAME", "I-NAME",
    "B-AGE", "I-AGE",
    "B-DATE", "I-DATE",
    "B-ORGANIZATION", "I-ORGANIZATION",
]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}


# ======================
# Hàm tokenize + align nhãn
# ======================
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# ======================
# Tokenize toàn bộ dataset
# ======================
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)


# ======================
# Khởi tạo model + Trainer
# ======================
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

args = TrainingArguments(
    output_dir="ner_vielectra_checkpoint",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
)

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return metric.compute(predictions=true_predictions, references=true_labels)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ======================
# Train model
# ======================
trainer.train()
