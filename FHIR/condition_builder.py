def build_fhir_condition_bundle(patient_data, ner_results):

    bundle = {
        "status": "SUCCESS",
        "fhir_bundle": {
            "resourceType": "Bundle",
            "entry": []
        }
    }

    disease_entities = [ent for ent in ner_results if ent.get("label") == "DISEASE"]

    past_medical_history = patient_data.get("medical_history", {}).get("past_medical_history")

    if past_medical_history:
        pmh_entities = [
            ent for ent in disease_entities 
            if ent.get("text") in past_medical_history 
        ]
        
        for ent in pmh_entities:
            pmh_linking = ent.get("linking", [{}])[0] 
            
            condition_resource = {
                "resource": {
                    "resourceType": "Condition",
                    "category": [{"coding": [{"code": "problem-list-item"}]}], 
                    "code": {
                        "coding": [
                            {
                                "system": "ICD-11",
                                "code": pmh_linking.get("ICD11_Code"),
                                "display": pmh_linking.get("ICD11_Title_EN") if pmh_linking.get("ICD11_Title_EN") else ent.get("text")
                            }
                        ],
                        "text": ent.get("text") 
                    }
                }
            }
            bundle["fhir_bundle"]["entry"].append(condition_resource)

    diagnosis = patient_data.get("diagnosis_text_input")

    if diagnosis:
        dx_entities = [
            ent for ent in disease_entities 
            if ent.get("text") in diagnosis
        ]

        for ent in dx_entities:
            dx_linking = ent.get("linking", [{}])[0] 

            condition_resource = {
                "resource": {
                    "resourceType": "Condition",
                    "category": [{"coding": [{"code": "encounter-diagnosis"}]}], 
                    "code": {
                        "coding": [
                            {
                                "system": "ICD-11",
                                "code": dx_linking.get("ICD11_Code"),
                                "display": dx_linking.get("ICD11_Title_EN") if dx_linking.get("ICD11_Title_EN") else ent.get("text")
                            }
                        ],
                        "text": ent.get("text") 
                    }
                }
            }
            bundle["fhir_bundle"]["entry"].append(condition_resource)

    return bundle