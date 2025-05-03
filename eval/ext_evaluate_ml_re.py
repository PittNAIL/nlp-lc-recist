#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate Relation Extraction (RE) performance on external data.

Usage Example:
--------------
python evaluate_re.py \
    --test_files data/test_files.txt \
    --model_dir ./models/ml_models/re_model \
    --output_dir results/re_evaluation

Assumptions:
------------
1. test_files.txt is a text file listing all annotated relation files.
2. The annotation includes columns for ent1_text, ent1_type, ent2_text, ent2_type,
   relation (the gold label: "Treatment_Response" or "No_Relation").
3. We load multiple classifiers from model_dir (random_forest, xgboost, etc.).
4. The script produces classification reports, confusion matrices, PR/ROC curves, etc.
"""

import os
import sys
import logging
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc, average_precision_score

import torch
from transformers import AutoTokenizer, AutoModel

# -------------------------------------------------------------------------
# Example loader function: adapt for your external data format.
# -------------------------------------------------------------------------
def load_data_from_files_with_relations(file_list):
    """
    Load relation-annotated data. 
    Return a DataFrame with columns:
      - ent1_text
      - ent1_type
      - ent2_text
      - ent2_type
      - relation
      - context (optional)
    Adjust as needed for your annotation format.
    """
    data = []
    for file_path in file_list:
        # Dummy example
        data.append({
            "file_name": os.path.basename(file_path),
            "ent1_text": "chemotherapy",
            "ent1_type": "Chemotherapy",
            "ent2_text": "partial response",
            "ent2_type": "Partial_Response",
            "relation": "Treatment_Response",
            "context": "the patient had partial response after chemo"
        })
    return pd.DataFrame(data)

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/re_evaluation.log"),
            logging.StreamHandler()
        ]
    )

def preprocess_data(df):
    """
    Combine ent1_text, context, ent2_text, ent1_type, ent2_type into a single string
    that the model can embed. Adjust as needed.
    """
    df['combined_text'] = (
        df['ent1_text'].fillna('') + ' ' +
        df['context'].fillna('') + ' ' +
        df['ent2_text'].fillna('') + ' ' +
        df['ent1_type'].fillna('') + ' ' +
        df['ent2_type'].fillna('')
    )
    return df

def encode_text_in_batches(text_list, tokenizer, model, batch_size=16, max_length=128):
    """
    Convert text_list into embeddings using the provided tokenizer + huggingface model.
    Returns a NumPy array of shape [N, hidden_size].
    """
    embeddings = []
    device = torch.device('cpu')
    model.eval().to(device)

    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i+batch_size]
            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            outputs = model(**encoded_input)
            # Use [CLS] embedding
            batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_emb)

    return np.vstack(embeddings)

def load_models(model_dir):
    """
    Load multiple classifiers (random_forest, xgboost, etc.) and label_encoder
    from the specified folder. Adjust the filenames if needed.
    """
    models = {}
    try:
        models['random_forest'] = joblib.load(os.path.join(model_dir, 're_random_forest_model.joblib'))
        models['xgboost'] = joblib.load(os.path.join(model_dir, 're_xgboost_model.joblib'))
        models['lightgbm'] = joblib.load(os.path.join(model_dir, 're_lightgbm_model.joblib'))
        models['adaboost'] = joblib.load(os.path.join(model_dir, 're_adaboost_model.joblib'))
        models['gradient_boosting'] = joblib.load(os.path.join(model_dir, 're_gradient_boosting_model.joblib'))
        models['svm'] = joblib.load(os.path.join(model_dir, 're_svm_model.joblib'))
        models['logistic_regression'] = joblib.load(os.path.join(model_dir, 're_logistic_regression_model.joblib'))
        models['label_encoder'] = joblib.load(os.path.join(model_dir, 're_label_encoder.joblib'))
    except Exception as e:
        logging.error(f"Error loading RE models: {str(e)}")
        raise
    return models

def evaluate_models(models, X_embeddings, y_true, label_encoder, df_processed, output_dir):
    """
    Evaluate each classifier in 'models' using the precomputed embeddings (X_embeddings)
    and the gold labels (y_true). Save classification reports, confusion matrices, PR/ROC, etc.
    """
    os.makedirs(output_dir, exist_ok=True)
    master_report_path = os.path.join(output_dir, 're_evaluation_reports.txt')

    y_classes = label_encoder.inverse_transform(y_true)
    class_labels = label_encoder.classes_

    with open(master_report_path, 'w') as master_report_file:
        for model_name, model_obj in models.items():
            if model_name == 'label_encoder':
                continue

            logging.info(f"Evaluating {model_name} model...")
            y_proba = model_obj.predict_proba(X_embeddings)

            # Find optimal threshold based on best F1
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba[:, 1])
            f1_scores = [2*(p*r)/(p+r) if (p+r) > 0 else 0 for p, r in zip(precision, recall)]
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

            y_pred_optimal = (y_proba[:, 1] >= optimal_threshold).astype(int)
            y_pred_labels = label_encoder.inverse_transform(y_pred_optimal)
            y_true_labels = label_encoder.inverse_transform(y_true)

            # Classification report
            report = classification_report(y_true_labels, y_pred_labels, zero_division=0)
            print(f"\nClassification Report - {model_name}:")
            print(report)

            master_report_file.write(f"=== {model_name} ===\n")
            master_report_file.write(f"Classification Report:\n{report}\n")
            master_report_file.write(f"Optimal Threshold: {optimal_threshold:.4f}\n")

            # Confusion Matrix
            cm = confusion_matrix(y_true_labels, y_pred_labels, labels=class_labels)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
            plt.title(f"Confusion Matrix - {model_name}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
            plt.savefig(cm_path)
            plt.close()

            # ROC Curve, AUC
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f"ROC Curve - {model_name}")
            plt.legend(loc="lower right")
            roc_path = os.path.join(output_dir, f"{model_name}_roc_curve.png")
            plt.savefig(roc_path)
            plt.close()

            # Precision-Recall Curve, AUC-PR
            average_precision = average_precision_score(y_true, y_proba[:, 1])
            plt.figure()
            plt.step(recall, precision, where='post', color='b', alpha=0.7)
            plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title(f'Precision-Recall curve: AP={average_precision:.2f} - {model_name}')
            pr_path = os.path.join(output_dir, f"{model_name}_pr_curve.png")
            plt.savefig(pr_path)
            plt.close()

            master_report_file.write(f"AUC-ROC: {roc_auc:.4f}\n")
            master_report_file.write(f"AUC-PR: {average_precision:.4f}\n\n")

            # Save misclassified samples
            misclassified_idx = np.where(y_pred_optimal != y_true)[0]
            misclassified = df_processed.iloc[misclassified_idx].copy()
            misclassified['true_label'] = y_true_labels[misclassified_idx]
            misclassified['predicted_label'] = y_pred_labels[misclassified_idx]
            csv_path = os.path.join(output_dir, f"{model_name}_misclassified_samples.csv")
            misclassified.to_csv(csv_path, index=False)

    logging.info("All model evaluations completed.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate RE pipeline on external data.")
    parser.add_argument("--test_files", required=True, help="Text file listing annotated relation files.")
    parser.add_argument("--model_dir", required=True, help="Directory containing RE classifiers + label_encoder.")
    parser.add_argument("--tokenizer_dir", default=None,
                        help="Directory or model name for the Hugging Face tokenizer (default: same as model_dir).")
    parser.add_argument("--output_dir", default="results/re_evaluation", help="Directory to save outputs.")
    args = parser.parse_args()

    setup_logging()
    logging.info("Starting Relation Extraction evaluation...")

    # 1. Load test file paths
    with open(args.test_files, 'r') as f:
        test_files = [line.strip() for line in f if line.strip()]

    if not test_files:
        logging.error("No test files found. Exiting.")
        sys.exit(1)

    # 2. Load annotated data
    df_relations = load_data_from_files_with_relations(test_files)
    if df_relations.empty:
        logging.error("No relation data found. Exiting.")
        sys.exit(1)

    # 3. Preprocess data
    df_processed = preprocess_data(df_relations)

    # 4. Prepare label arrays
    y_test_labels = df_processed['relation'].values

    # 5. Load classifiers + label encoder
    models = load_models(args.model_dir)
    label_encoder = models['label_encoder']
    y_test = label_encoder.transform(y_test_labels)

    # 6. Load huggingface model for embeddings
    tokenizer_dir = args.tokenizer_dir if args.tokenizer_dir else args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    hf_model = AutoModel.from_pretrained(args.model_dir).cpu()

    # 7. Convert text to embeddings
    X_test_embeddings = encode_text_in_batches(df_processed['combined_text'].tolist(), tokenizer, hf_model)

    # 8. Evaluate
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate_models(models, X_test_embeddings, y_test, label_encoder, df_processed, args.output_dir)

    logging.info("Relation Extraction evaluation completed successfully.")

if __name__ == "__main__":
    main()
