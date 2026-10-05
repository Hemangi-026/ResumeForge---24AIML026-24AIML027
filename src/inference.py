import os
import sys
import json
import logging
from typing import Dict, Any, List, Union
import numpy as np
import joblib
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pdf_extractor import extract_text_from_pdf
from src.preprocessing import clean_resume_text, extract_skills_from_text
from src.models_dl import ResumeBiLSTM
from src.features import texts_to_sequences

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResumeClassifierPipeline:
    """
    Production-ready end-to-end inference pipeline supporting both
    Word2Vec + BiLSTM (Deep Learning) and Calibrated Linear SVM (Classical ML).
    """
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Label Encoder
        le_path = os.path.join(models_dir, "label_encoder.joblib")
        if not os.path.exists(le_path):
            raise FileNotFoundError(f"Label encoder not found at {le_path}")
        self.label_encoder = joblib.load(le_path)
        self.classes = list(self.label_encoder.classes_)
        
        # Load Classical Model & TF-IDF
        self.tfidf_vec = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.joblib"))
        self.classical_model = joblib.load(os.path.join(models_dir, "linear_svm.joblib"))
        
        # Load DL Model & Word2Vec mapping if available
        self.dl_model = None
        self.word2idx = None
        word2idx_path = os.path.join(models_dir, "word2idx.json")
        bilstm_path = os.path.join(models_dir, "best_bilstm.pt")
        
        if os.path.exists(word2idx_path) and os.path.exists(bilstm_path):
            with open(word2idx_path, "r", encoding="utf-8") as f:
                self.word2idx = json.load(f)
            
            # Dummy embedding matrix shape: [vocab_size, 150]
            vocab_size = len(self.word2idx)
            dummy_matrix = np.zeros((vocab_size, 150), dtype=np.float32)
            self.dl_model = ResumeBiLSTM(dummy_matrix, num_classes=len(self.classes)).to(self.device)
            state_dict = torch.load(bilstm_path, map_location=self.device)
            self.dl_model.load_state_dict(state_dict)
            self.dl_model.eval()
            logger.info("Successfully loaded PyTorch BiLSTM model.")

    def predict(
        self,
        source: Union[str, bytes],
        is_pdf: bool = False,
        model_type: str = "dl"
    ) -> Dict[str, Any]:
        """
        Unified prediction endpoint.
        source: File path, byte stream, or raw text.
        is_pdf: If True, invokes PDF text extraction first.
        model_type: 'dl' (Word2Vec + BiLSTM) or 'classical' (Linear SVM).
        """
        # 1. Extract text if PDF
        if is_pdf:
            raw_text = extract_text_from_pdf(source)
        else:
            raw_text = source if isinstance(source, str) else source.decode('utf-8', errors='ignore')

        raw_char_count = len(raw_text)
        raw_word_count = len(raw_text.split())

        if raw_word_count < 5:
            return {
                'success': False,
                'error': 'Resume text is too short or empty for classification.',
                'raw_text': raw_text
            }

        # 2. Domain-aware preprocessing (preserving tech terms)
        cleaned_text = clean_resume_text(raw_text)
        extracted_skills = extract_skills_from_text(raw_text)

        # 3. Model Inference
        if model_type == "dl" and self.dl_model is not None and self.word2idx is not None:
            seq = texts_to_sequences([cleaned_text], self.word2idx, max_len=300)
            tensor_x = torch.tensor(seq, dtype=torch.long).to(self.device)
            with torch.no_grad():
                logits = self.dl_model(tensor_x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        else:
            vec = self.tfidf_vec.transform([cleaned_text])
            probs = self.classical_model.predict_proba(vec)[0]

        top_indices = probs.argsort()[::-1]
        best_idx = top_indices[0]
        predicted_category = self.classes[best_idx]
        confidence = float(probs[best_idx])

        top_3 = []
        for idx in top_indices[:3]:
            top_3.append({
                'category': self.classes[idx],
                'probability': round(float(probs[idx]) * 100, 2)
            })

        return {
            'success': True,
            'predicted_category': predicted_category,
            'confidence': round(confidence * 100, 2),
            'top_3_predictions': top_3,
            'extracted_skills': extracted_skills,
            'raw_word_count': raw_word_count,
            'raw_char_count': raw_char_count,
            'model_used': "Word2Vec + BiLSTM (Deep Learning)" if model_type == "dl" else "Linear SVM (Classical TF-IDF)",
            'cleaned_text_preview': " ".join(cleaned_text.split()[:40]) + "..."
        }

if __name__ == "__main__":
    pipeline = ResumeClassifierPipeline()
    sample = (
        "Senior Software Engineer with 6 years experience in Python, C++, SQL, AWS, "
        "and machine learning. Designed microservices and NLP pipelines using PyTorch and Docker."
    )
    res_dl = pipeline.predict(sample, model_type="dl")
    print("\n--- Test DL Inference ---")
    print(f"Predicted: {res_dl['predicted_category']} ({res_dl['confidence']}%)")
    print(f"Top 3: {res_dl['top_3_predictions']}")
    print(f"Skills: {res_dl['extracted_skills']}")

    res_svm = pipeline.predict(sample, model_type="classical")
    print("\n--- Test Classical SVM Inference ---")
    print(f"Predicted: {res_svm['predicted_category']} ({res_svm['confidence']}%)")
    print(f"Top 3: {res_svm['top_3_predictions']}")
