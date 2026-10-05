import os
import sys
import json
import logging
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.features import (
    split_data,
    build_tfidf_features,
    train_word2vec_embeddings,
    build_embedding_matrix,
    texts_to_sequences
)
from src.models_classical import train_classical_models
from src.models_dl import train_dl_model
from src.evaluate import generate_comparison_table, plot_confusion_matrix, perform_error_analysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=================================================================")
    logger.info("  STARTING SAMATRIX RESUMEFORGE 2026 MODEL TRAINING PIPELINE    ")
    logger.info("=================================================================")

    cleaned_csv = os.path.join("data", "processed", "cleaned_resumes.csv")
    if not os.path.exists(cleaned_csv):
        logger.error(f"Cleaned dataset not found at {cleaned_csv}. Run src/eda.py first.")
        return

    df = pd.read_csv(cleaned_csv)
    logger.info(f"Loaded {len(df)} cleaned resumes across {df['category'].nunique()} categories.")

    # 1. Stratified Split (70/15/15)
    split_dict = split_data(df, text_col='cleaned_text', label_col='category', train_size=0.70, val_size=0.15, test_size=0.15, random_state=42)
    X_train, y_train = split_dict['X_train'], split_dict['y_train']
    X_val, y_val = split_dict['X_val'], split_dict['y_val']
    X_test, y_test = split_dict['X_test'], split_dict['y_test']
    label_encoder = split_dict['label_encoder']
    class_names = list(label_encoder.classes_)

    # Save Label Encoder and Class Names
    os.makedirs("models", exist_ok=True)
    joblib.dump(label_encoder, "models/label_encoder.joblib")
    with open("models/classes.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    logger.info("Saved Label Encoder to models/label_encoder.joblib")

    # 2. Stage 6: TF-IDF Features (Fitted ONLY on training data)
    X_train_vec, X_val_vec, X_test_vec, tfidf_vec = build_tfidf_features(
        X_train, X_val, X_test,
        ngram_range=(1, 2),
        max_features=8000,
        min_df=2,
        max_df=0.75,
        sublinear_tf=True
    )
    joblib.dump(tfidf_vec, "models/tfidf_vectorizer.joblib")
    logger.info("Saved TF-IDF Vectorizer to models/tfidf_vectorizer.joblib")

    # 3. Stage 6: Train Classical ML Baselines
    logger.info("--- Training Classical Models ---")
    classical_models, classical_metrics, classical_preds, classical_probs = train_classical_models(
        X_train_vec, y_train, X_val_vec, y_val, X_test_vec, y_test
    )
    
    # Save individual classical models
    for m_name, m_obj in classical_models.items():
        fname = m_name.lower().replace(' ', '_') + ".joblib"
        joblib.dump(m_obj, os.path.join("models", fname))
        logger.info(f"Saved {m_name} to models/{fname}")

    # 4. Stage 7: Deep Learning (Word2Vec + BiLSTM)
    logger.info("--- Training Word2Vec Embeddings ---")
    w2v_model = train_word2vec_embeddings(list(X_train), vector_size=150, window=5, min_count=2, epochs=12)
    w2v_model.save("models/word2vec.model")
    
    embedding_matrix, word2idx = build_embedding_matrix(w2v_model, embedding_dim=150)
    with open("models/word2idx.json", "w", encoding="utf-8") as f:
        json.dump(word2idx, f)
        
    X_train_seq = texts_to_sequences(list(X_train), word2idx, max_len=300)
    X_val_seq = texts_to_sequences(list(X_val), word2idx, max_len=300)
    X_test_seq = texts_to_sequences(list(X_test), word2idx, max_len=300)

    logger.info("--- Training PyTorch BiLSTM Neural Network ---")
    dl_model, dl_metrics, dl_preds, dl_probs = train_dl_model(
        X_train_seq, y_train,
        X_val_seq, y_val,
        X_test_seq, y_test,
        embedding_matrix=embedding_matrix,
        num_classes=len(class_names),
        batch_size=64,
        epochs=10
    )

    # 5. Stage 8: Model Benchmarking & Comparison Table
    logger.info("--- Compiling Comparative Benchmark ---")
    all_results = dict(classical_metrics)
    all_results['Word2Vec + BiLSTM'] = dl_metrics
    df_comparison = generate_comparison_table(all_results, "models/model_comparison.csv")

    # Determine Best Model
    best_model_name = df_comparison.iloc[0]['model']
    logger.info(f"🏆 Best Performing Model: {best_model_name}")

    if best_model_name in classical_models:
        best_model = classical_models[best_model_name]
        best_preds = classical_preds[best_model_name]
        best_probs = classical_probs[best_model_name]
        joblib.dump(best_model, "models/best_model.joblib")
        logger.info(f"Saved best model ({best_model_name}) to models/best_model.joblib")
    else:
        best_preds = dl_preds
        best_probs = dl_probs
        with open("models/best_model_type.txt", "w") as f:
            f.write("dl")
        # Still keep linear svm as fallback best classical model
        joblib.dump(classical_models['Linear SVM'], "models/best_model.joblib")

    # Stage 8: Confusion Matrix for Best Model
    plot_confusion_matrix(y_test, best_preds, class_names, best_model_name, "models/confusion_matrix.png")

    # Stage 9: Error Analysis
    logger.info("--- Performing In-Depth Error Analysis ---")
    err_df = perform_error_analysis(
        list(X_test), y_test, best_preds, best_probs, class_names, "models/error_analysis.csv"
    )

    # Save sample test inputs for verification and demo
    sample_resumes = []
    for i in range(min(10, len(X_test))):
        sample_resumes.append({
            'text': str(X_test[i]),
            'true_category': class_names[y_test[i]],
            'predicted_category': class_names[best_preds[i]],
            'confidence': float(best_probs[i][best_preds[i]])
        })
    with open("models/sample_test_resumes.json", "w", encoding="utf-8") as f:
        json.dump(sample_resumes, f, indent=2)

    logger.info("=================================================================")
    logger.info("  TRAINING PIPELINE COMPLETED SUCCESSFULLY!                     ")
    logger.info("=================================================================")

if __name__ == "__main__":
    main()
