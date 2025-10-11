from flask import Flask, request, jsonify
from NER.ner_module import extract_entities
from EntityLinking.entity_linking import link_entity

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text", "")

    # NER
    entities = extract_entities(text)

    # Entity Linking
    for e in entities:
        linked = link_entity(e["text"], top_k=1)
        e["linked"] = linked

    return jsonify({
        "input": text,
        "results": entities
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)

