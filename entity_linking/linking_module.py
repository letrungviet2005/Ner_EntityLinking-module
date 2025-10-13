from sentence_transformers import SentenceTransformer, util
import json
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

with open(os.path.join(BASE_DIR, "drug_db_linking
                       .json"), "r", encoding="utf-8") as f:
    DRUG_DB = json.load(f)

def link_entity(entity_text: str, top_k=1):
    emb_entity = sbert.encode(entity_text, convert_to_tensor=True)
    emb_db = sbert.encode([item["name"] for item in DRUG_DB], convert_to_tensor=True)

    cos_scores = util.cos_sim(emb_entity, emb_db)[0]
    top_idx = cos_scores.argsort(descending=True)[:top_k]

    linked = []
    for idx in top_idx:
        score = float(cos_scores[idx])
        item = DRUG_DB[idx]
        linked.append({
            "name": item["name"],
            "id": item["id"],
            "description": item["description"],
            "score": score
        })
    return linked
