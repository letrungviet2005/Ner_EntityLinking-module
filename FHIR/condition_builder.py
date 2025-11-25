def build_fhir_condition_bundle(patient_data, ner_results):
    """
    Xây dựng FHIR Bundle cho:
      - Bệnh nền (past_medical_history)
      - Chẩn đoán hiện tại (diagnosis_text_input)
    """

    bundle = {
        "status": "SUCCESS",
        "fhir_bundle": {
            "resourceType": "Bundle",
            "entry": []
        }
    }

    past_medical_history = patient_data.get("medical_history", {}).get("past_medical_history")
    if past_medical_history:
        disease_entity = {
            "text": past_medical_history,
            "label": "DISEASE",
            "linking": []  
        }
        if ner_results:
            for entity in ner_results:
                if entity.get("text") == past_medical_history and entity.get("linking"):
                    disease_entity["linking"] = entity["linking"]

        linking = disease_entity.get("linking", [])
        condition_resource = {
            "resource": {
                "resourceType": "Condition",
                "category": [{"coding": [{"code": "problem-list-item"}]}],  # Bệnh nền
                "code": {
                    "coding": [
                        {
                            "system": "ICD-11",
                            "code": linking[0].get("ICD11_Code") if linking else None,
                            "display": linking[0].get("ICD11_Title_EN") if linking else past_medical_history
                        }
                    ],
                    "text": past_medical_history
                }
            }
        }
        bundle["fhir_bundle"]["entry"].append(condition_resource)

    for entity in ner_results:
        if entity.get("label") == "DISEASE":
            if entity.get("text") == past_medical_history:
                continue

            linking = entity.get("linking", [])
            condition_resource = {
                "resource": {
                    "resourceType": "Condition",
                    "category": [{"coding": [{"code": "encounter-diagnosis"}]}], 
                    "code": {
                        "coding": [
                            {
                                "system": "ICD-11",
                                "code": linking[0].get("ICD11_Code") if linking else None,
                                "display": linking[0].get("ICD11_Title_EN") if linking else entity.get("text")
                            }
                        ],
                        "text": entity.get("text")
                    }
                }
            }
            bundle["fhir_bundle"]["entry"].append(condition_resource)

    return bundle
