from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ======================
# Load mô hình & tokenizer
# ======================
MODEL_NAME = "D:/2025/AI-FOR-LIFE-2025/module_medical_ner_linking/checkpoint/ner_vielectra_checkpoint"

print("🔹 Đang tải mô hình, vui lòng chờ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"  
)

text = "Bácsĩ Nguyễn Trung Nguyên Giám đốc Trung tâm Chống độc Bệnh viện Bạch Mai cho biết bệnh nhân được chuyển đến bệnh viện ngày 7/3 chẩn đoán ngộ độc thuốc điều trị sốt rét chloroquine"


results = ner_pipeline(text)

print("\n===== KẾT QUẢ NHẬN DIỆN THỰC THỂ =====")
for r in results:
    print(f"🟢 Từ: {r['word']}\t| Nhãn: {r['entity_group']}\t| Độ tin cậy: {r['score']:.4f}")
