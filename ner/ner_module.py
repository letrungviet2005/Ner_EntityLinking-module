import json
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_name = "NlpHUST/ner-vietnamese-electra-base"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


with open(os.path.join(BASE_DIR, "drug_db.json"), "r", encoding="utf-8") as f:
    DRUG_LIST = [item["name"] for item in json.load(f)]

def extract_entities(text: str):
    results = ner_pipeline(text)
    entities = []

    for r in results:
        entities.append({
            "text": r["word"],
            "label": r["entity_group"],
            "score": float(r["score"])
        })

    for drug in DRUG_LIST:
        if drug.lower() in text.lower():
            entities.append({
                "text": drug,
                "label": "DRUG",
                "score": 1.0
            })

    return entities
