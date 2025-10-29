import json
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mô hình tiếng Việt
MODEL_NAME_VI = "NlpHUST/ner-vietnamese-electra-base"

# Mô hình tiếng Anh chuyên nhận diện thuốc
MODEL_NAME_EN = "d4data/biomedical-ner-all"

DB_PATH = os.path.join(BASE_DIR, "drug_db.json")

# Load mô hình tiếng Việt
tokenizer_vi = AutoTokenizer.from_pretrained(MODEL_NAME_VI, use_fast=False)
model_vi = AutoModelForTokenClassification.from_pretrained(MODEL_NAME_VI)
ner_vi_pipeline = pipeline(
    "ner",
    model=model_vi,
    tokenizer=tokenizer_vi,
    aggregation_strategy="simple"
)

# Load mô hình tiếng Anh
tokenizer_en = AutoTokenizer.from_pretrained(MODEL_NAME_EN)
model_en = AutoModelForTokenClassification.from_pretrained(MODEL_NAME_EN)
ner_en_pipeline = pipeline(
    "ner",
    model=model_en,
    tokenizer=tokenizer_en,
    aggregation_strategy="simple"
)

# Load danh sách thuốc từ file JSON
with open(DB_PATH, "r", encoding="utf-8") as f:
    DRUG_LIST = [item["name"] for item in json.load(f)]


def extract_drugs_en(text: str):
    """Nhận diện thuốc bằng mô hình tiếng Anh."""
    results = ner_en_pipeline(text)
    entities = []
    for r in results:
        label = r["entity_group"].upper()
        if "DRUG" in label or "CHEM" in label:
            entities.append({
                "text": r["word"],
                "label": "DRUG",
                "score": float(r["score"])
            })
    return entities


def extract_entities(text: str):
    """Kết hợp mô hình tiếng Việt, tiếng Anh và database thuốc."""
    # 1. Nhận diện bằng mô hình tiếng Việt
    vi_results = ner_vi_pipeline(text)
    entities = []

    for r in vi_results:
        entities.append({
            "text": r["word"],
            "label": r["entity_group"],
            "score": float(r["score"])
        })

    # 2. Nhận diện thuốc bằng mô hình tiếng Anh
    en_drug_entities = extract_drugs_en(text)
    entities.extend(en_drug_entities)

    # 3. Nhận diện thuốc từ file JSON
    for drug in DRUG_LIST:
        if drug.lower() in text.lower():
            entities.append({
                "text": drug,
                "label": "DRUG",
                "score": 1.0
            })

    # 4. Loại trùng lặp
    unique_entities = []
    seen = set()
    for ent in entities:
        key = (ent["text"].lower(), ent["label"])
        if key not in seen:
            seen.add(key)
            unique_entities.append(ent)

    return unique_entities


