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
except ImportError as e:
    # Xử lý trường hợp không thể import (ví dụ: thiếu file __init__.py hoặc sai đường dẫn)
    print(f"LỖI IMPORT: Không thể tải module. Vui lòng kiểm tra cấu trúc thư mục và file __init__.py. Chi tiết: {e}")
    # Nếu lỗi xảy ra, bạn nên thay thế các hàm này bằng hàm giả (mock) để ứng dụng Flask không sập
    def extract_entities(text): return []
    def link_icd_entity(entity_text, top_k): return [{"error": "Linking module failed to load."}]


app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    # --- KHẮC PHỤC LỖI NAMERROR TẠI ĐÂY ---
    # Lấy dữ liệu JSON từ request
    data = request.json
    # Trích xuất trường 'text' từ dữ liệu, biến này là nguyên nhân gây lỗi NameError
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Vui lòng cung cấp trường 'text' trong yêu cầu."}), 400

    # 1. Bước 1: Trích xuất thực thể (NER)
    try:
        entities = extract_entities(text)
    except Exception as e:
        # Trả về lỗi 500 nếu quá trình trích xuất thất bại
        return jsonify({"error": f"Lỗi trong quá trình trích xuất thực thể (NER): {str(e)}"}), 500

    linked_results = []

    # 2. Bước 2: Lặp và Ánh xạ thực thể (Linking)
    for entity in entities:
        # Dùng "text" thay vì "entity" (đã sửa để khớp với JSON mẫu của bạn)
        entity_text = entity.get("text") 
        
        # Thêm các trường cơ bản từ NER (score, label, text)
        result_item = {
            "text": entity_text,
            "label": entity.get("label"),
            "score": entity.get("score"), # Giữ lại score từ NER
        }

        # Chỉ thực hiện linking nếu có văn bản thực thể
        if entity_text:
            try:
                # Gọi hàm linking
                linked_data = link_icd_entity(entity_text, top_k=1) 
                result_item["linking"] = linked_data # Thêm kết quả ánh xạ
            except Exception as e:
                # Nếu linking thất bại, vẫn trả về kết quả NER và ghi lại lỗi
                result_item["linking"] = [{"error": f"Lỗi linking: {str(e)}"}]
        else:
            # Trường hợp thực thể không có text, bỏ qua linking
            result_item["linking"] = []
            
        linked_results.append(result_item)
            
    # 3. Trả về kết quả cuối cùng
    return jsonify({
        "input": text,
        "results": linked_results
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)