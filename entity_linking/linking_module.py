from sentence_transformers import SentenceTransformer, util
import json
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "drug_db_linking.json")

sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

with open(DB_PATH, "r", encoding="utf-8") as f:
    DRUG_DB = json.load(f)

DB_NAMES = [item["name"] for item in DRUG_DB]
DB_EMBEDDINGS = sbert.encode(DB_NAMES, convert_to_tensor=True, show_progress_bar=False)

def link_entity(entity_text: str, top_k: int = 1):

    emb_entity = sbert.encode(entity_text, convert_to_tensor=True)

    cos_scores = util.cos_sim(emb_entity, DB_EMBEDDINGS)[0]

    top_idx = torch.topk(cos_scores, k=top_k)

    linked = []
    for score, idx in zip(top_idx.values, top_idx.indices):
        item = DRUG_DB[int(idx)]
        linked.append({
            "name": item["name"],
            "id": item.get("id", None),
            "description": item.get("description", ""),
            "score": float(score)
        })

    return linked


