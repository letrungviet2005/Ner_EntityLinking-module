import json
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME_VI = "D:/2025/AI-FOR-LIFE-2025/module_medical_ner_linking/checkpoints/ner_icd11_final" 

try:
    print(f"Đang tải mô hình: {MODEL_NAME_VI}")
    tokenizer_vi = AutoTokenizer.from_pretrained(MODEL_NAME_VI, use_fast=False)
    model_vi = AutoModelForTokenClassification.from_pretrained(MODEL_NAME_VI)
    ner_vi_pipeline = pipeline(
        "ner",
        model=model_vi,
        tokenizer=tokenizer_vi,
        aggregation_strategy="simple"
    )
except Exception as e:
    print(f"Lỗi khi tải mô hình tiếng Việt: {e}")
    ner_vi_pipeline = None


def extract_entities(text: str):

    if not ner_vi_pipeline:
        print("Mô hình tiếng Việt chưa được tải thành công. Không thể thực hiện nhận diện.")
        return []

    vi_results = ner_vi_pipeline(text)
    entities = []

    for r in vi_results:
        entities.append({
            "text": r["word"],
            "label": r["entity_group"],
            "score": float(r["score"])
        })

    unique_entities = []
    seen = set()
    for ent in entities:
        key = (ent["text"].lower(), ent["label"]) 
        if key not in seen:
            seen.add(key)
            unique_entities.append(ent)

    return unique_entities