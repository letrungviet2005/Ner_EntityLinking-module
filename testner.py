from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_NAME = "D:/2025/AI-FOR-LIFE-2025/module_medical_ner_linking/checkpoint/checkpoint"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

# text = "Bác sĩ Nguyễn Trung Nguyên Giám đốc Trung tâm Chống độc Bệnh viện Bạch Mai cho biết bệnh nhân được chuyển đến bệnh viện ngày 7/3 chẩn đoán ngộ độc thuốc điều trị sốt rét chloroquine"
text = "Bệnh nhân Trần Thị Lan nhập viện ngày 12/4 với triệu chứng nôn mửa, đau bụng và được chỉ định truyền dịch, uống than hoạt tính."
# text = "Ông Nguyễn Văn Hùng 45 tuổi đến Bệnh viện Chợ Rẫy trong tình trạng sốt cao 39 độ, đau đầu và ho khan kéo dài 5 ngày."
# text = "Bệnh nhân nam Nguyễn Minh Tuấn được bác sĩ kê đơn thuốc paracetamol và amoxicillin để điều trị cảm cúm và viêm họng cấp"
# text = "Ngày 15/5, tại Bệnh viện 108, bệnh nhân nữ Lê Thị Thu được chẩn đoán mắc viêm phổi do vi khuẩn và đang được dùng ceftriaxone"
# text = "Bác sĩ Phạm Văn Dũng cho biết bệnh nhân có biểu hiện chóng mặt, đau ngực sau khi sử dụng thuốc giảm đau ibuprofen quá liều"
# text = "Tại Bệnh viện Nhi Trung ương, bé gái 8 tuổi được chẩn đoán mắc tay chân miệng và đang điều trị bằng thuốc kháng virus acyclovir"
# text = "Ông Lê Văn Quý nhập viện ngày 2/6 trong tình trạng khó thở, mệt mỏi, được xác định nhiễm viêm phổi do COVID-19"
# text = "Bệnh nhân Nguyễn Thanh Bình sau khi uống thuốc ngủ diazepam liều cao có dấu hiệu buồn nôn và mất ý thức tạm thời"
# text = "Bác sĩ Đỗ Thị Hạnh, Bệnh viện Đại học Y Dược TP.HCM, chia sẻ bệnh nhân bị dị ứng thuốc kháng sinh ciprofloxacin."
# text = "Ngày 10/8, tại Bệnh viện Hữu nghị Việt Đức, nam bệnh nhân 32 tuổi được phẫu thuật cấp cứu sau khi ngộ độc thuốc diệt cỏ paraquat."
# text = "Nữ bệnh nhân Trần Ngọc Bích xuất hiện triệu chứng đau khớp, sốt nhẹ và được kê thuốc prednisolone trong 7 ngày."
# text = "Ông Phan Văn Dũng đến khám tại Trung tâm Y tế quận Ba Đình vì đau đầu, mất ngủ và được kê đơn thuốc melatonin."
# text = "Bệnh nhân Nguyễn Thị Mai điều trị tiểu đường type 2 bằng metformin và insulin tại Bệnh viện Nội tiết Trung ương."
# text = "Ngày 3/9, bác sĩ Nguyễn Đức Long cho biết một bệnh nhân nam 50 tuổi bị xuất huyết tiêu hóa do dùng aspirin kéo dài."
# text = "Tại Bệnh viện Phạm Ngọc Thạch, bệnh nhân được chẩn đoán lao phổi và đang điều trị bằng rifampicin, isoniazid và pyrazinamide."
# text = "Bác sĩ Trần Quang Huy cho biết bệnh nhân bị viêm gan B mãn tính đang sử dụng tenofovir để kiểm soát virus."
# text = "Ông Lưu Văn Phước nhập viện ngày 22/7 trong tình trạng đau bụng dữ dội, được chẩn đoán viêm tụy cấp do rượu."
# text = "Bé trai Nguyễn Hữu Khang 10 tuổi bị sốt xuất huyết Dengue, đang được truyền dịch và theo dõi tại Bệnh viện Nhi Đồng 1."
# text = "Ngày 28/10, Bệnh viện Bạch Mai tiếp nhận một bệnh nhân ngộ độc rượu methanol, được lọc máu khẩn cấp để loại bỏ độc chất."


results = ner_pipeline(text)

for r in results:
    print(f"🟢 Từ: {r['word']}\t| Nhãn: {r['entity_group']}\t| Độ tin cậy: {r['score']:.4f}")
