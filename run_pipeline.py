import json
import yaml
import re
import joblib
import torch
from transformers import AutoTokenizer, AutoModel

def load_ner_patterns(pattern_path):
    """Load and compile NER regex patterns from a YAML file."""
    with open(pattern_path, 'r') as f:
        patterns = yaml.safe_load(f)
    return {entity: re.compile(pattern, re.I) for entity, pattern in patterns.items()}

def rule_based_ner(text, patterns):
    """Extract entities using rule-based regex matching."""
    entities = []
    for ent_type, pattern in patterns.items():
        for match in pattern.finditer(text):
            start, end = match.span()
            entities.append({
                "text": text[start:end],
                "type": ent_type,
                "start": start,
                "end": end,
                "confidence": 1.0  # Rule-based extractions have fixed confidence
            })
    return entities

def load_re_model(model_dir):
    """
    Load the relation extraction components: a tokenizer, the BioClinicalBERT model,
    and a logistic regression classifier saved via joblib.
    """
    tokenizer = AutoTokenizer.from_pretrained((f"{model_dir}/tokenizer"))
    model = AutoModel.from_pretrained((f"{model_dir}/model"))
    classifier = joblib.load(f"{model_dir}/re_logistic_regression_model.joblib")
    return tokenizer, model, classifier

def extract_relations(entities, tokenizer, model, classifier):
    """
    For each treatment-response entity pair, create an input string, generate an embedding,
    and use the classifier to decide if a relation exists (if probability > 0.5).
    """
    # Define treatment and response entity types
    treatment_types = [
        'Cancer_Surgery', 'Radiotherapy', 'Chemotherapy',
        'Immunotherapy', 'Targeted_Therapy'
    ]
    response_types = [
        'Complete_Response', 'Partial_Response',
        'Stable_Disease', 'Progressive_Disease'
    ]
    
    treatments = [e for e in entities if e['type'] in treatment_types]
    responses = [e for e in entities if e['type'] in response_types]
    relations = []
    
    for treatment in treatments:
        for response in responses:
            # Concatenate texts of treatment and response with a separator
            relation_text = f"{treatment['text']} [SEP] {response['text']}"
            inputs = tokenizer(relation_text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            # Use the [CLS] token embedding as the sentence representation
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
            proba = classifier.predict_proba(embedding)[:, 1][0]
            if proba > 0.5:
                relations.append({
                    "treatment": treatment['text'],
                    "response": response['text'],
                    "confidence": float(proba)
                })
    return relations

def run_pipeline(dummy_text, ner_pattern_path, re_model_dir):
    # 1. Load the rule-based NER patterns
    patterns = load_ner_patterns(ner_pattern_path)
    # 2. Extract entities using the rule-based approach
    entities = rule_based_ner(dummy_text, patterns)
    # 3. Load the relation extraction model components
    tokenizer, model, classifier = load_re_model(re_model_dir)
    # 4. Extract relationships between treatment and response entities
    relations = extract_relations(entities, tokenizer, model, classifier)
    
    # Combine results into one dictionary
    result = {
        "text": dummy_text,
        "entities": entities,
        "relations": relations
    }
    return result

if __name__ == '__main__':
    # Create a dummy clinical note
    dummy_note = (
        "The patient underwent a left upper lobectomy and received adjuvant chemotherapy with paclitaxel. "
        "Following the treatment, a complete response was observed on imaging. "
        "There was no evidence of progressive disease."
    )
    
    # Define paths (adjust if necessary)
    ner_pattern_file = "./models/Final_models_to_deploy/rule_based_models/rule_based_ner_patterns.yaml"
    re_model_directory = "./models/Final_models_to_deploy/ml_models/re_model/bioclinicalbert_ml_models"
    
    # Run the pipeline
    results = run_pipeline(dummy_note, ner_pattern_file, re_model_directory)
    
    # Print out the results in a nicely formatted JSON structure
    print(json.dumps(results, indent=2))
