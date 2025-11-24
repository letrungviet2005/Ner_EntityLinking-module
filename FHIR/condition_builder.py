def create_fhir_condition(entity, patient_id="example-patient-id"):
    return {
        "resourceType": "Condition",
        "id": f"cond-{entity['text'].replace(' ', '-')}",
        "clinicalStatus": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": "active"
                }
            ]
        },
        "verificationStatus": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                    "code": "confirmed"
                }
            ]
        },
        "code": {
            "text": entity["text"]
        },
        "subject": {
            "reference": f"Patient/{patient_id}"
        },
        "note": [
            {
                "text": f"Score từ mô hình NER: {entity['score']}"
            }
        ]
    }
def create_fhir_observation(entity, patient_id="example-patient-id"):
    return {
        "resourceType": "Observation",
        "id": f"obs-{entity['text'].replace(' ', '-')}",
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "clinical",
                        "display": "Clinical"
                    }
                ]
            }
        ],
        "code": {
            "text": entity["text"]
        },
        "subject": {
            "reference": f"Patient/{patient_id}"
        },
        "note": [
            {
                "text": f"Score từ mô hình NER: {entity['score']}"
            }
        ]
    }

def entity_to_fhir(entity, patient_id="example-patient-id"):
    label = entity["label"].upper()

    if label in ["SYMPTOM"]:
        return create_fhir_observation(entity, patient_id)

    if label in ["DISEASE", "CONDITION"]:
        return create_fhir_condition(entity, patient_id)


    return None

