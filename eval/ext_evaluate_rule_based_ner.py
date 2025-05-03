#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate Named Entity Recognition (NER) performance on external data.

Usage Example:
--------------
python evaluate_ner.py \
    --test_files data/test_files.txt \
    --patterns ./Final_models_to_deploy/rule_based_models/rule_based_ner_patterns.yaml \
    --output_dir results/ner_evaluation

Assumptions:
------------
1. test_files.txt is a text file listing all annotated XML (or other) files.
2. The annotation includes "entity_type" for each span in the gold standard.
3. The script will run rule-based classification, compare to gold labels,
   and produce classification report, confusion matrix, error analysis, etc.

You can adapt `load_xml_data` to handle your annotation format if needed.
"""

import os
import sys
import yaml
import re
import logging
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------------------------------------------------
# Example loader function: you can replace or modify this for your data.
# -------------------------------------------------------------------------
def load_xml_data(file_list):
    """
    Load and parse gold-standard annotated data from XML or another format.
    Return a pandas DataFrame with at least these columns:
      - text (the raw text)
      - entity_type (the gold label)
    In your real code, also parse offsets, attributes, etc. if needed.
    """
    data = []
    for file_path in file_list:
        # Dummy placeholder logic:
        # In practice, parse your XML or JSON, extracting text + entity info
        # The below is just an example row per file
        data.append({
            "file_name": os.path.basename(file_path),
            "text": "Example text from " + os.path.basename(file_path),
            "entity_type": "Chemotherapy",  # Example
        })
    return pd.DataFrame(data)

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/ner_evaluation.log"),
            logging.StreamHandler()
        ]
    )

def load_patterns(patterns_path):
    """
    Load a YAML file with your regex patterns and compile them.
    """
    with open(patterns_path, 'r') as f:
        patterns = yaml.safe_load(f)
    return {entity_type: re.compile(pattern, re.IGNORECASE) for entity_type, pattern in patterns.items()}

def classify_text(text, pattern_dict):
    """
    Classify text as a single entity type or "Other" using rule-based patterns.
    If multiple patterns match, the first match is returned in this example.
    You can adapt logic for multi-label or more complex matching.
    """
    for entity_type, pattern in pattern_dict.items():
        if pattern.search(text):
            # Example: if we find a match, return that entity_type
            return entity_type
    return "Other"

def main():
    parser = argparse.ArgumentParser(description="Evaluate rule-based NER on external data.")
    parser.add_argument("--test_files", required=True, help="Path to a text file listing all test annotation files.")
    parser.add_argument("--patterns", required=True, help="Path to the rule-based NER patterns YAML.")
    parser.add_argument("--output_dir", default="results/ner_evaluation", help="Directory to save evaluation outputs.")
    args = parser.parse_args()

    setup_logging()
    logging.info("Starting NER evaluation...")

    # 1. Load test file paths
    with open(args.test_files, 'r') as f:
        test_files = [line.strip() for line in f if line.strip()]

    if not test_files:
        logging.error("No test files found. Exiting.")
        sys.exit(1)

    # 2. Load test data from your annotation
    df_test = load_xml_data(test_files)
    logging.info(f"Loaded test data with shape: {df_test.shape}")

    # 3. Load patterns
    pattern_dict = load_patterns(args.patterns)

    # 4. Predict entity type for each row
    df_test["predicted_entity_type"] = df_test["text"].apply(
        lambda x: classify_text(x, pattern_dict)
    )

    # 5. Evaluate entity-level performance
    os.makedirs(args.output_dir, exist_ok=True)
    report_file = os.path.join(args.output_dir, 'classification_report.txt')

    true_labels = df_test["entity_type"]
    pred_labels = df_test["predicted_entity_type"]

    report = classification_report(true_labels, pred_labels, zero_division=0)
    with open(report_file, 'w') as f:
        f.write(report)

    print("\nEntity-Level Classification Report:")
    print(report)
    logging.info(f"Classification report saved to {report_file}")

    # 6. Confusion Matrix
    entity_types = sorted(df_test["entity_type"].unique())
    cm = confusion_matrix(true_labels, pred_labels, labels=entity_types)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=entity_types, yticklabels=entity_types, cmap='Blues')
    plt.title("Entity-Level Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    logging.info(f"Confusion matrix saved to {cm_path}")

    # 7. Optional: Error analysis (false positives/negatives)
    df_test['correct'] = (df_test['entity_type'] == df_test['predicted_entity_type'])
    false_positives = df_test[(df_test['predicted_entity_type'] != 'Other') &
                              (df_test['predicted_entity_type'] != df_test['entity_type'])]
    false_negatives = df_test[(df_test['entity_type'] != 'Other') &
                              (df_test['predicted_entity_type'] == 'Other')]

    false_positives.to_csv(os.path.join(args.output_dir, 'false_positives.csv'), index=False)
    false_negatives.to_csv(os.path.join(args.output_dir, 'false_negatives.csv'), index=False)
    logging.info("Error analysis CSVs saved.")

    logging.info("NER evaluation completed successfully.")

if __name__ == "__main__":
    main()
