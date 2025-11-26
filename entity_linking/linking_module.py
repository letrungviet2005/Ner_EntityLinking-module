from sentence_transformers import SentenceTransformer, util
import json
import os
import torch
import numpy as np 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICD_DB_PATH = os.path.join(BASE_DIR, "data/icd11v2.json") 

sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

with open(ICD_DB_PATH, "r", encoding="utf-8") as f:
    ICD_DB = json.load(f)

ICD_DB = [item for item in ICD_DB if item.get("ICD11_Title_EN_VN") and item["ICD11_Title_EN_VN"] != "NaN"]

DB_NAMES_VN = [item["ICD11_Title_EN_VN"] for item in ICD_DB]
DB_EMBEDDINGS = sbert.encode(DB_NAMES_VN, convert_to_tensor=True, show_progress_bar=False)

def link_icd_entity(entity_text: str, top_k: int = 1):

    emb_entity = sbert.encode(entity_text, convert_to_tensor=True)

    cos_scores = util.cos_sim(emb_entity, DB_EMBEDDINGS)[0]

    top_idx = torch.topk(cos_scores, k=top_k)

    linked = []
    for score, idx in zip(top_idx.values, top_idx.indices):
        item = ICD_DB[int(idx)]
        
        linked.append({
            "ICD11_Title_EN": item.get("ICD11_Title_EN"),
            "ICD11_Title_VN": item.get("ICD11_Title_EN_VN"),
            "ICD11_Code": item.get("ICD11_Code", None),
            "Score": float(score)
        })

    return linked

