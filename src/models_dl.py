import os
import sys
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models_classical import evaluate_predictions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResumeBiLSTM(nn.Module):
    """
    Bidirectional LSTM Classifier with Word2Vec Embeddings and Dual-Pooling.
    """
    def __init__(
        self,
        embedding_matrix: np.ndarray,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 24,
        dropout: float = 0.35
    ):
        super(ResumeBiLSTM, self).__init__()
        num_embeddings, embedding_dim = embedding_matrix.shape
        
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=False,
            padding_idx=0
        )
        
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Dual pooling (Max + Average) produces (hidden_dim * 2) * 2 = hidden_dim * 4
        feature_dim = hidden_dim * 4
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len]
        mask = (x != 0).unsqueeze(-1)  # [batch_size, seq_len, 1]
        emb = self.embedding(x)        # [batch_size, seq_len, emb_dim]
        
        lstm_out, _ = self.bilstm(emb) # [batch_size, seq_len, hidden_dim * 2]
        
        # Apply padding mask for pooling
        masked_out = lstm_out.masked_fill(~mask, -1e9)
        max_pool = torch.max(masked_out, dim=1)[0]
        
        lengths = mask.sum(dim=1).clamp(min=1)
        avg_pool = (lstm_out * mask).sum(dim=1) / lengths
        
        pooled = torch.cat([max_pool, avg_pool], dim=1) # [batch_size, feature_dim]
        logits = self.fc(pooled)                        # [batch_size, num_classes]
        return logits

def train_dl_model(
    X_train_seq: np.ndarray,
    y_train: np.ndarray,
    X_val_seq: np.ndarray,
    y_val: np.ndarray,
    X_test_seq: np.ndarray,
    y_test: np.ndarray,
    embedding_matrix: np.ndarray,
    num_classes: int = 24,
    batch_size: int = 64,
    epochs: int = 15,
    device: Optional[str] = None,
    save_path: str = os.path.join("models", "best_bilstm.pt")
) -> Tuple[nn.Module, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Trains BiLSTM deep learning model with early stopping and class-weighted loss.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training Deep Learning model on device: {device}")

    # Compute class weights for imbalanced classes
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    # PyTorch DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train_seq, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val_seq, dtype=torch.long), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_seq, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = ResumeBiLSTM(embedding_matrix, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_y)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch_y.cpu().numpy())

        val_loss /= len(val_ds)
        val_acc = (np.array(val_preds) == np.array(val_targets)).mean()
        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            torch.save(best_weights, save_path)

    # Load best checkpoint for test evaluation
    if best_weights is not None:
        model.load_state_dict(best_weights)
        logger.info(f"Restored best checkpoint with Val Loss: {best_val_loss:.4f}")

    # Test set evaluation
    model.eval()
    test_preds, test_probs = [], []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            test_preds.extend(preds)
            test_probs.append(probs)

    test_preds_np = np.array(test_preds)
    test_probs_np = np.vstack(test_probs)

    metrics = evaluate_predictions(y_test, test_preds_np, "Word2Vec + BiLSTM")
    return model, metrics, test_preds_np, test_probs_np
