"""
8.4) LSTM
==============
LSTM classifier in PyTorch that consumes windowed sequences of features.
Wraps the PyTorch module behind the BaseModel interface for seamless
integration with the CPCV pipeline.
"""

import copy
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.cpcv.models.base import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_WINDOW = 21            # lookback window (~1 trading month)
LSTM_BATCH_SIZE = 64
LSTM_EPOCHS = 100
LSTM_LR = 1e-3
LSTM_PATIENCE = 10          # early stopping patience


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------
def create_sequences(
    X, y=None, w=None, window: int = LSTM_WINDOW
) -> tuple:
    """Reshape 2D features into 3D windowed sequences.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Feature matrix, shape (T, n_features).
    y : np.ndarray or pd.Series, optional
        Labels. If None, prediction mode.
    w : np.ndarray or pd.Series, optional
        Sample weights.
    window : int
        Lookback window length.

    Returns
    -------
    Training mode (y provided):
        (X_seq, y_seq, w_seq, valid_indices)
    Prediction mode (y is None):
        (X_seq, valid_indices)

    X_seq has shape (T - window + 1, window, n_features).
    valid_indices maps each sequence to its original positional index.
    """
    X_arr = X.values if hasattr(X, "values") else X
    T, n_feat = X_arr.shape

    n_seq = T - window + 1
    if n_seq <= 0:
        raise ValueError(
            f"Input length {T} is shorter than window {window}. "
            "Cannot create sequences."
        )

    X_seq = np.zeros((n_seq, window, n_feat), dtype=np.float32)
    for i in range(n_seq):
        X_seq[i] = X_arr[i : i + window]

    valid_indices = np.arange(window - 1, T)

    if y is None:
        return X_seq, valid_indices

    # training mode: align y and w
    y_arr = y.values if hasattr(y, "values") else y
    y_seq = y_arr[valid_indices]

    w_seq = None
    if w is not None:
        w_arr = w.values if hasattr(w, "values") else w
        w_seq = w_arr[valid_indices]

    return X_seq, y_seq, w_seq, valid_indices


# ---------------------------------------------------------------------------
# PyTorch module
# ---------------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    """LSTM network with a linear classification head."""

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, window, n_features).

        Returns
        -------
        torch.Tensor
            Raw logits, shape (batch, n_classes). No softmax applied.
        """
        # lstm_out: (batch, window, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]                   # (batch, hidden_size)
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)            # (batch, n_classes)
        return logits


# ---------------------------------------------------------------------------
# BaseModel wrapper
# ---------------------------------------------------------------------------
class LSTMModel(BaseModel):
    """LSTM classifier wrapped in the BaseModel interface."""

    def __init__(self, n_features: int, n_classes: int = 2, seed: int = 42):
        super().__init__(n_features, n_classes, seed)
        self.net = None
        self.window = LSTM_WINDOW
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_valid_indices = None

    def fit(
        self,
        X_train,
        y_train,
        sample_weight=None,
        X_val=None,
        y_val=None,
    ) -> None:
        # ── reproducibility ───────────────────────────────────────────
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # ── build sequences ───────────────────────────────────────────
        X_seq, y_seq, w_seq, _ = create_sequences(
            X_train, y_train, sample_weight, window=self.window
        )

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_seq, y_val_seq, _, _ = create_sequences(
                X_val, y_val, window=self.window
            )

        # ── tensors ───────────────────────────────────────────────────
        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_seq, dtype=torch.long).to(self.device)
        w_t = (
            torch.tensor(w_seq, dtype=torch.float32).to(self.device)
            if w_seq is not None
            else torch.ones(len(y_seq), dtype=torch.float32).to(self.device)
        )

        if has_val:
            X_val_t = torch.tensor(X_val_seq, dtype=torch.float32).to(self.device)
            y_val_t = torch.tensor(y_val_seq, dtype=torch.long).to(self.device)

        # ── dataloader ────────────────────────────────────────────────
        train_ds = TensorDataset(X_t, y_t, w_t)
        train_dl = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True)

        # ── model ─────────────────────────────────────────────────────
        self.net = LSTMClassifier(
            n_features=self.n_features, n_classes=self.n_classes
        ).to(self.device)

        # ── class weights ─────────────────────────────────────────────
        class_counts = np.bincount(y_seq, minlength=self.n_classes)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / class_weights.sum() * self.n_classes
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        # ── loss, optimizer, scheduler ────────────────────────────────
        criterion = nn.CrossEntropyLoss(weight=class_weights_t, reduction="none")
        optimizer = torch.optim.Adam(self.net.parameters(), lr=LSTM_LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        # ── training loop ─────────────────────────────────────────────
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(LSTM_EPOCHS):
            self.net.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_b, y_b, w_b in train_dl:
                optimizer.zero_grad()
                logits = self.net(X_b)
                per_sample_loss = criterion(logits, y_b)
                # integrate AFML sample weights
                weighted_loss = (per_sample_loss * w_b).mean()
                weighted_loss.backward()
                optimizer.step()

                epoch_loss += weighted_loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)

            # ── validation ────────────────────────────────────────────
            if has_val:
                self.net.eval()
                with torch.no_grad():
                    val_logits = self.net(X_val_t)
                    val_loss = nn.CrossEntropyLoss(weight=class_weights_t)(
                        val_logits, y_val_t
                    ).item()

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.net.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= LSTM_PATIENCE:
                    logger.info(
                        "LSTM early stopping at epoch %d (best val loss: %.4f).",
                        epoch + 1, best_val_loss,
                    )
                    break
            else:
                scheduler.step(avg_train_loss)

        # restore best weights
        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.net.eval()
        logger.info(
            "LSTM fitted: %d sequences, %d epochs completed, device=%s.",
            len(y_seq), epoch + 1, self.device,
        )

    def predict_proba(self, X) -> np.ndarray:
        X_seq, valid_indices = create_sequences(X, window=self.window)
        self.last_valid_indices = valid_indices

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

        self.net.eval()
        with torch.no_grad():
            logits = self.net(X_t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()

        return proba

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_logits(self, X) -> np.ndarray:
        """Return raw pre-softmax logits."""
        X_seq, valid_indices = create_sequences(X, window=self.window)
        self.last_valid_indices = valid_indices

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

        self.net.eval()
        with torch.no_grad():
            logits = self.net(X_t).cpu().numpy()

        return logits

    def get_name(self) -> str:
        return "LSTM"