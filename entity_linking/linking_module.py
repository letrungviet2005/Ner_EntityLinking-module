from sentence_transformers import SentenceTransformer, util
import json
import os
import torch
import numpy as np 
from typing import List, Dict

# --- Cấu hình và Tải dữ liệu ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Đảm bảo ICD_DB_PATH chính xác
ICD_DB_PATH = os.path.join(BASE_DIR, "data/icd11v3.json") 

sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

with open(ICD_DB_PATH, "r", encoding="utf-8") as f:
    ICD_DB = json.load(f)

ICD_DB = [item for item in ICD_DB if item.get("ICD11_Title_EN_VN") and item["ICD11_Title_EN_VN"] != "NaN"]

DB_NAMES_VN = [item["ICD11_Title_EN_VN"] for item in ICD_DB]

DB_EMBEDDINGS = sbert.encode(DB_NAMES_VN, convert_to_tensor=True, show_progress_bar=False)


def link_icd_entity(entity_text: str, top_k: int = 1) -> List[Dict]:


    entity_text_lower = entity_text.strip().lower()

    candidate_indices = []
    candidate_items = []
    
    for idx, item in enumerate(ICD_DB):
        icd_name_vn_lower = item["ICD11_Title_EN_VN"].lower()
        if entity_text_lower in icd_name_vn_lower:
             candidate_indices.append(idx)
             candidate_items.append(item)

    if not candidate_indices:
        print(f"DEBUG: Không tìm thấy ứng viên substring cho '{entity_text}'. Quay về tìm kiếm toàn bộ.")
        candidate_indices = list(range(len(ICD_DB)))
        candidate_embeddings = DB_EMBEDDINGS
        candidate_items = ICD_DB
    else:

        candidate_embeddings = DB_EMBEDDINGS[candidate_indices]

    emb_entity = sbert.encode(entity_text, convert_to_tensor=True)

    if candidate_embeddings.dim() == 1:
        cos_scores = util.cos_sim(emb_entity, candidate_embeddings.unsqueeze(0))[0]
    else:
        cos_scores = util.cos_sim(emb_entity, candidate_embeddings)[0]

    top_k_to_use = min(top_k, len(candidate_items))
    top_idx = torch.topk(cos_scores, k=top_k_to_use)

    linked = []
    for score, idx in zip(top_idx.values, top_idx.indices):
        original_idx = candidate_indices[int(idx)]

        item = ICD_DB[original_idx] 
        linked.append({
            "ICD11_Title_EN": item.get("ICD11_Title_EN"),
            "ICD11_Title_VN": item.get("ICD11_Title_EN_VN"),
            "ICD11_Code": item.get("ICD11_Code", None),
            "Score": float(score)
        })

    return linked