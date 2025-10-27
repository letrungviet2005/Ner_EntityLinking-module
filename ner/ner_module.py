import json
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "NlpHUST/ner-vietnamese-electra-base"
DB_PATH = os.path.join(BASE_DIR, "drug_db.json")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

with open(DB_PATH, "r", encoding="utf-8") as f:
    DRUG_LIST = [item["name"] for item in json.load(f)]

def extract_entities(text: str):
    model_results = ner_pipeline(text)
    entities = []

    for r in model_results:
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


