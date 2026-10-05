import os
import sys
import logging
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_comparison_table(results_dict: Dict[str, Dict[str, float]], save_path: str = "models/model_comparison.csv") -> pd.DataFrame:
    """
    Constructs and saves the comparative performance table.
    """
    rows = list(results_dict.values())
    df_comp = pd.DataFrame(rows)
    df_comp = df_comp.sort_values(by='macro_f1', ascending=False).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_comp.to_csv(save_path, index=False)
    logger.info("Model Comparison Benchmark Table:")
    logger.info("\n" + df_comp.to_string(index=False))
    return df_comp

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_path: str = "models/confusion_matrix.png"
):
    """
    Generates and saves the 24x24 multi-class confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Normalize by true label support
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm_norm,
        annot=False,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Recall'}
    )
    plt.title(f"Normalized Confusion Matrix — {model_name} (24 Classes)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Predicted Category", fontsize=12, fontweight='semibold')
    plt.ylabel("Actual Category", fontsize=12, fontweight='semibold')
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {save_path}")

def perform_error_analysis(
    X_test_texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    save_path: str = "models/error_analysis.csv"
) -> pd.DataFrame:
    """
    Inspects wrongly classified resumes and categorizes error reasons.
    """
    errors = []
    for idx in range(len(y_true)):
        true_idx = y_true[idx]
        pred_idx = y_pred[idx]
        if true_idx != pred_idx:
            conf = float(probs[idx][pred_idx]) if probs is not None else 0.0
            actual_cat = class_names[true_idx]
            pred_cat = class_names[pred_idx]
            raw_text = X_test_texts[idx]
            
            # Diagnose reason
            words = raw_text.split()
            word_count = len(words)
            if word_count < 40:
                reason = "Sparse / Short Resume: Insufficient domain-specific terms"
            elif {actual_cat, pred_cat}.issubset({'BUSINESS-DEVELOPMENT', 'CONSULTANT', 'SALES', 'PUBLIC-RELATIONS'}):
                reason = "Semantic Overlap: Commercial / Client-facing role ambiguity"
            elif {actual_cat, pred_cat}.issubset({'INFORMATION-TECHNOLOGY', 'ENGINEERING', 'DIGITAL-MEDIA'}):
                reason = "Technical Overlap: Shared engineering & software stack keywords"
            elif {actual_cat, pred_cat}.issubset({'FINANCE', 'ACCOUNTANT', 'BANKING'}):
                reason = "Financial Domain Overlap: Common ledger, audit, and banking vocabulary"
            else:
                reason = "Cross-Functional Role: Candidate possesses hybrid inter-disciplinary experience"

            errors.append({
                'actual_category': actual_cat,
                'predicted_category': pred_cat,
                'confidence': round(conf, 4),
                'word_count': word_count,
                'primary_reason': reason,
                'text_preview': " ".join(words[:45]) + "..."
            })

    err_df = pd.DataFrame(errors)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    err_df.to_csv(save_path, index=False)
    logger.info(f"Total misclassifications on test set: {len(err_df)} / {len(y_true)} ({len(err_df)/len(y_true)*100:.2f}%)")
    logger.info("Error categories breakdown:")
    logger.info("\n" + str(err_df['primary_reason'].value_counts()))
    return err_df
