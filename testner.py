from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_NAME = "D:/2025/AI-FOR-LIFE-2025/module_medical_ner_linking/checkpoints/ner_icd11_final_v4/ner_icd11_final_v4"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

text = "nhiễm trùng đường ruột, u ác tính"


results = ner_pipeline(text)

for r in results:
    print(f"🟢 Từ: {r['word']}\t| Nhãn: {r['entity_group']}\t| Độ tin cậy: {r['score']:.4f}")
