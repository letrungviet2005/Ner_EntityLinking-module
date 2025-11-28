def build_fhir_condition_bundle(patient_data, ner_results):

    bundle = {
        "status": "SUCCESS",
        "fhir_bundle": {
            "resourceType": "Bundle",
            "entry": []
        }
    }

    # ===== 1) PAST MEDICAL HISTORY =====
    past_medical_history = patient_data.get("medical_history", {}).get("past_medical_history")

    if past_medical_history:
        pmh_linking = None
        for ent in ner_results:
            if ent.get("label") == "DISEASE" and ent.get("text") in past_medical_history:
                if ent.get("linking"):
                    pmh_linking = ent["linking"][0]
                    break

        condition_resource = {
            "resource": {
                "resourceType": "Condition",
                "category": [{"coding": [{"code": "problem-list-item"}]}],
                "code": {
                    "coding": [
                        {
                            "system": "ICD-11",
                            "code": pmh_linking.get("ICD11_Code") if pmh_linking else None,
                            "display": pmh_linking.get("ICD11_Title_EN") if pmh_linking else past_medical_history
                        }
                    ],
                    "text": past_medical_history
                }
            }
        }
        bundle["fhir_bundle"]["entry"].append(condition_resource)

    # ===== 2) CURRENT DIAGNOSIS =====
    diagnosis = patient_data.get("diagnosis_text_input")

    if diagnosis:
        dx_linking = None
        for ent in ner_results:
            if ent.get("label") == "DISEASE" and ent.get("text") == diagnosis:
                if ent.get("linking"):
                    dx_linking = ent["linking"][0]
                    break

        condition_resource = {
            "resource": {
                "resourceType": "Condition",
                "category": [{"coding": [{"code": "encounter-diagnosis"}]}],
                "code": {
                    "coding": [
                        {
                            "system": "ICD-11",
                            "code": dx_linking.get("ICD11_Code") if dx_linking else None,
                            "display": dx_linking.get("ICD11_Title_EN") if dx_linking else diagnosis
                        }
                    ],
                    "text": diagnosis
                }
            }
        }
        bundle["fhir_bundle"]["entry"].append(condition_resource)

    return bundle
