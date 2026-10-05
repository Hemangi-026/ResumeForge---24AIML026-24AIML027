import os
import sys
import logging
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_data(
    df: pd.DataFrame,
    text_col: str = 'cleaned_text',
    label_col: str = 'category',
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Stratified split into Train (70%), Validation (15%), and Test (15%).
    Strictly guarantees identical class distributions and zero data leakage.
    Converts to standard numpy arrays to avoid arrow array indexing conflicts.
    """
    assert abs((train_size + val_size + test_size) - 1.0) < 1e-5, "Split fractions must sum to 1.0"
    
    # 1. Encode labels
    label_encoder = LabelEncoder()
    y_encoded = np.array(label_encoder.fit_transform(df[label_col]), dtype=np.int64)
    X_raw = np.array(df[text_col].astype(str).tolist(), dtype=object)
    
    # First split: Train vs Temp (Val + Test)
    temp_size = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_raw,
        y_encoded,
        test_size=temp_size,
        stratify=y_encoded,
        random_state=random_state
    )
    
    # Second split: Val vs Test
    test_fraction_of_temp = test_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_fraction_of_temp,
        stratify=y_temp,
        random_state=random_state
    )
    
    logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'label_encoder': label_encoder
    }

def build_tfidf_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 8000,
    min_df: int = 2,
    max_df: float = 0.75,
    sublinear_tf: bool = True
) -> Tuple[Any, Any, Any, TfidfVectorizer]:
    """
    Fits TF-IDF strictly on the training set, transforming Val and Test.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf
    )
    logger.info("Fitting TF-IDF vectorizer on training split...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    logger.info(f"TF-IDF matrix shape: {X_train_vec.shape}, Vocabulary size: {len(vectorizer.vocabulary_)}")
    return X_train_vec, X_val_vec, X_test_vec, vectorizer

def train_word2vec_embeddings(
    corpus_texts: List[str],
    vector_size: int = 150,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 15
) -> Word2Vec:
    """
    Trains continuous Word2Vec skip-gram embeddings on training texts.
    """
    tokenized_sentences = [str(text).split() for text in corpus_texts]
    logger.info(f"Training Word2Vec on {len(tokenized_sentences)} training documents...")
    w2v_model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
        sg=1  # Skip-gram
    )
    logger.info(f"Word2Vec vocabulary size: {len(w2v_model.wv)}")
    return w2v_model

def texts_to_sequences(
    texts: List[str],
    word2idx: Dict[str, int],
    max_len: int = 300
) -> np.ndarray:
    """
    Converts list of text documents into fixed-length padded integer sequences.
    0 index reserved for PAD, 1 for UNK.
    """
    sequences = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, text in enumerate(texts):
        tokens = str(text).split()
        seq = [word2idx.get(w, 1) for w in tokens[:max_len]]
        sequences[i, :len(seq)] = seq
    return sequences

def build_embedding_matrix(
    w2v_model: Word2Vec,
    embedding_dim: int = 150
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Extracts PyTorch-compatible embedding weights matrix and token index mapping.
    """
    vocab = list(w2v_model.wv.index_to_key)
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for idx, word in enumerate(vocab, start=2):
        word2idx[word] = idx
        
    matrix_len = len(word2idx)
    embedding_matrix = np.zeros((matrix_len, embedding_dim), dtype=np.float32)
    embedding_matrix[1] = np.random.normal(scale=0.6, size=(embedding_dim,))
    
    for word, idx in word2idx.items():
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
            
    return embedding_matrix, word2idx
