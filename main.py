from flask import Flask, request, jsonify
import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from ner.ner_module import extract_entities
    from entity_linking.linking_module import link_icd_entity
    from FHIR.condition_builder import build_fhir_condition_bundle
except ImportError as e:
    def extract_entities(text): return []
    def link_icd_entity(entity_text, top_k): return [{"error": "Linking module failed to load."}]
    def build_fhir_condition_bundle(input_text, ner_results): return {"status": "FAIL", "message": "FHIR builder not available"}

app = Flask(__name__)
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json

    if not data:
        return jsonify({"error": "Vui lòng gửi JSON trong request body."}), 400

    diagnosis_text = data.get("diagnosis_text_input", "")
    past_medical_history = data.get("medical_history", {}).get("past_medical_history", "")

    if not diagnosis_text and not past_medical_history:
        return jsonify({"error": "Không có dữ liệu cho 'diagnosis_text_input' hoặc 'past_medical_history'."}), 400

    texts_to_process = []
    if past_medical_history:
        texts_to_process.append(past_medical_history)
    if diagnosis_text:
        texts_to_process.append(diagnosis_text)

    all_entities = []

    for text in texts_to_process:
        try:
            entities = extract_entities(text)
        except Exception as e:
            return jsonify({"error": f"Lỗi NER với text '{text}': {str(e)}"}), 500

        for entity in entities:
            entity_text = entity.get("text")
            if entity_text:
                try:
                    linked_data = link_icd_entity(entity_text, top_k=1)
                    entity["linking"] = linked_data
                except Exception as e:
                    entity["linking"] = [{"error": f"Lỗi linking: {str(e)}"}]
            else:
                entity["linking"] = []

        all_entities.extend(entities)

    fhir_bundle = build_fhir_condition_bundle(data, all_entities)
    return jsonify(fhir_bundle)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
