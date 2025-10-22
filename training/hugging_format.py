from datasets import Dataset
import json

with open("ner_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Mỗi item gồm: text + entities
texts = [d["text"] for d in data]
labels = [d["entities"] for d in data]

dataset = Dataset.from_dict({"text": texts, "entities": labels})
dataset.save_to_disk("ner_dataset_hf")
