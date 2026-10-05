import os
import sys
import logging
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Computes comprehensive multi-class metrics: Accuracy, Macro-F1, Weighted-F1, Precision, Recall.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    logger.info(f"[{model_name}] Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f} | Weighted-F1: {weighted_f1:.4f}")
    return {
        'model': model_name,
        'accuracy': round(acc, 4),
        'macro_f1': round(macro_f1, 4),
        'weighted_f1': round(weighted_f1, 4),
        'macro_precision': round(macro_prec, 4),
        'macro_recall': round(macro_rec, 4)
    }

def train_classical_models(
    X_train_vec: Any,
    y_train: np.ndarray,
    X_val_vec: Any,
    y_val: np.ndarray,
    X_test_vec: Any,
    y_test: np.ndarray
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    """
    Trains and benchmarks:
    1. Multinomial Naive Bayes
    2. Logistic Regression (balanced)
    3. Calibrated Linear Support Vector Classifier (balanced)
    """
    models = {
        'Multinomial Naive Bayes': MultinomialNB(alpha=0.1),
        'Logistic Regression': LogisticRegression(
            C=1.5,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        'Linear SVM': CalibratedClassifierCV(
            LinearSVC(C=0.8, class_weight='balanced', random_state=42, max_iter=2000),
            cv=3
        )
    }
    
    results = {}
    test_predictions = {}
    test_probabilities = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train_vec, y_train)
        
        # Validation performance
        val_pred = model.predict(X_val_vec)
        logger.info(f"Validation metrics for {name}:")
        evaluate_predictions(y_val, val_pred, f"{name} (Val)")
        
        # Test performance
        test_pred = model.predict(X_test_vec)
        test_predictions[name] = test_pred
        
        if hasattr(model, 'predict_proba'):
            test_probabilities[name] = model.predict_proba(X_test_vec)
        else:
            test_probabilities[name] = None
            
        metrics = evaluate_predictions(y_test, test_pred, name)
        results[name] = metrics

    return models, results, test_predictions, test_probabilities
