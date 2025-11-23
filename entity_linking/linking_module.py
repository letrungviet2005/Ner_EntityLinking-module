from sentence_transformers import SentenceTransformer, util
import json
import os
import torch
import numpy as np # Cần để xử lý giá trị NaN nếu có

# --- Cài đặt cơ bản ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Đổi tên file DB cho phù hợp với dữ liệu ICD
ICD_DB_PATH = os.path.join(BASE_DIR, "icd11v2.json") 

sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# --- 1. Tải và chuẩn bị DB ICD-11 ---
with open(ICD_DB_PATH, "r", encoding="utf-8") as f:
    ICD_DB = json.load(f)

# Lọc các mục không hợp lệ (ví dụ: thiếu ICD11_Title_EN_VN hoặc NaN)
# Lưu ý: Nếu file JSON của bạn chứa giá trị "NaN" (chữ), cần xử lý bằng tay
ICD_DB = [item for item in ICD_DB if item.get("ICD11_Title_EN_VN") and item["ICD11_Title_EN_VN"] != "NaN"]


# Lấy Tên tiếng Việt để nhúng (Embedding)
DB_NAMES_VN = [item["ICD11_Title_EN_VN"] for item in ICD_DB]
DB_EMBEDDINGS = sbert.encode(DB_NAMES_VN, convert_to_tensor=True, show_progress_bar=False)

# --- 2. Hàm ánh xạ thực thể ---
def link_icd_entity(entity_text: str, top_k: int = 1):
    """
    Ánh xạ thực thể được trích xuất (ví dụ: từ NER) sang ICD-11 dựa trên độ tương đồng ngữ nghĩa.
    
    Args:
        entity_text (str): Văn bản thực thể (ví dụ: tên bệnh) được NER trích xuất.
        top_k (int): Số lượng kết quả có độ tương đồng cao nhất cần trả về.
        
    Returns:
        list: Danh sách các kết quả ánh xạ ICD-11.
    """
    
    # Nhúng (Embed) thực thể đầu vào
    emb_entity = sbert.encode(entity_text, convert_to_tensor=True)

    # Tính toán độ tương đồng Cosine
    cos_scores = util.cos_sim(emb_entity, DB_EMBEDDINGS)[0]

    # Lấy k chỉ mục có điểm số cao nhất
    top_idx = torch.topk(cos_scores, k=top_k)

    linked = []
    for score, idx in zip(top_idx.values, top_idx.indices):
        item = ICD_DB[int(idx)]
        
        # Sửa lại cấu trúc trả về để lấy các trường thông tin ICD-11 cần thiết
        linked.append({
            "ICD11_Title_VN": item.get("ICD11_Title_EN_VN"), # <--- Thông tin bạn cần
            "ICD11_Code": item.get("ICD11_Code", None),
            "ICD11_Title_EN": item.get("ICD11_Title_EN", ""),
            "Score": float(score)
        })

    return linked

# --- Ví dụ kiểm tra ---
# Giả sử NER trích xuất được "bệnh tả"
entity_from_ner = "bệnh tả"
results = link_icd_entity(entity_from_ner, top_k=3)

print(f"Ánh xạ cho '{entity_from_ner}':")
for result in results:
    print(f"- Mã: {result['ICD11_Code']}, Tên VN: {result['ICD11_Title_VN']}, Điểm: {result['Score']:.4f}")