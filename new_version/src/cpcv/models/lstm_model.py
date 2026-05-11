"""
10.4) LSTM
==============
LSTM classifier in PyTorch, consumed via the BaseModel interface and trained
with a standard regularisation stack: tanh input normalisation, LayerNorm on
the pooled hidden state, gradient clipping (max-norm 1.0), label smoothing
(0.1), AdamW with cosine annealing + warm restarts.

Pooling: last-hidden-state pooling on a 14-day window (~2 BTC weeks, matched
to the 10-day TBL horizon). Short sequences plus modest training samples make
attention pooling unprofitable here, so the standard last-hidden-state
representation is preferred.
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

# --- Module-level constants -------------------------------------------------
# Hidden-size search space is [16, 32]; the notebook overrides per-split via tuning.
LSTM_HIDDEN_SIZE = 32
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
# 14-day window matches the TBL horizon and keeps parameter burden modest.
LSTM_WINDOW = 14
LSTM_BATCH_SIZE = 64
LSTM_EPOCHS = 100
LSTM_LR = 1e-3
LSTM_PATIENCE = 15
LSTM_LABEL_SMOOTHING = 0.1
LSTM_GRAD_CLIP_NORM = 1.0

# Warm-restart period T_0=25 sits below patience×2 so at least one restart can fire
# before early stopping could terminate training.
LSTM_WARMRESTART_T0 = 25
LSTM_WARMRESTART_TMULT = 2


# --- 1. Sequence construction ----------------------------------------------
# Reshape a 2D feature matrix into 3D windowed sequences for LSTM consumption.
def create_sequences(
    X, y=None, w=None, window: int = LSTM_WINDOW
) -> tuple:
    """Reshape ``(T, n_features)`` into windowed ``(T - window + 1, window, n_features)``.

    Training mode (``y`` provided) returns ``(X_seq, y_seq, w_seq, valid_indices)``;
    prediction mode returns ``(X_seq, valid_indices)``. ``valid_indices`` maps
    each sequence to the original positional index of its last timestep, which
    the caller uses to align predictions with the source timestamps.
    """
    X_arr = X.values if hasattr(X, "values") else X
    T, n_feat = X_arr.shape

    n_seq = T - window + 1
    if n_seq <= 0:
        raise ValueError(
            f"Input length {T} is shorter than window {window}. "
            "Cannot create sequences."
        )

    # Build sequences by sliding the window; final-timestep index drives label alignment.
    X_seq = np.zeros((n_seq, window, n_feat), dtype=np.float32)
    for i in range(n_seq):
        X_seq[i] = X_arr[i : i + window]

    valid_indices = np.arange(window - 1, T)

    if y is None:
        return X_seq, valid_indices

    # Training mode: align y (and optional w) to each sequence's last-timestep index.
    y_arr = y.values if hasattr(y, "values") else y
    y_seq = y_arr[valid_indices]

    w_seq = None
    if w is not None:
        w_arr = w.values if hasattr(w, "values") else w
        w_seq = w_arr[valid_indices]

    return X_seq, y_seq, w_seq, valid_indices


# --- 2. PyTorch module ------------------------------------------------------
# Plain nn.Module: LSTM stack → LayerNorm → dropout → linear head, returning raw logits.
class LSTMClassifier(nn.Module):
    """LSTM with last-hidden-state pooling and a dense classification head."""

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        hidden_size: int | None = None,
        num_layers: int | None = None,
        dropout: float | None = None,
    ):
        super().__init__()
        # Read constants at call time so tuning overrides via ``lstm_mod.LSTM_HIDDEN_SIZE = ...`` are picked up.
        hidden_size = LSTM_HIDDEN_SIZE if hidden_size is None else hidden_size
        num_layers = LSTM_NUM_LAYERS if num_layers is None else num_layers
        dropout = LSTM_DROPOUT if dropout is None else dropout

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, n_classes)

    # Forward pass returns raw logits (no softmax); caller handles probability conversion.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the LSTM and return ``(batch, n_classes)`` logits."""
        # lstm_out: (B, window, H) — per-timestep outputs.
        # h_n:    (num_layers, B, H) — final hidden state per layer.
        lstm_out, (h_n, _) = self.lstm(x)

        # Use the last layer's final hidden state as the sequence representation.
        context = h_n[-1]                                        # (B, H)

        context = self.layer_norm(context)
        context = self.dropout(context)
        logits = self.fc(context)                                # (B, n_classes)
        return logits


# --- 3. BaseModel wrapper ---------------------------------------------------
# Bridge between the CPCV pipeline (BaseModel contract) and the PyTorch LSTM stack.
class LSTMModel(BaseModel):
    """LSTM classifier wrapped in the BaseModel interface for the CPCV pipeline."""

    def __init__(self, n_features: int, n_classes: int = 2, seed: int = 42):
        super().__init__(n_features, n_classes, seed)
        self.net = None
        self.window = LSTM_WINDOW
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_valid_indices = None
        # Tanh-normalisation parameters fitted on training data only.
        self._input_mean = None
        self._input_std = None

    # Fit tanh normalisation parameters on train; tanh squashes fat-tailed features into [-1, 1].
    def _fit_input_norm(self, X: np.ndarray) -> None:
        """Fit ``z = tanh((x - mean) / (std + eps))`` parameters on training data."""
        self._input_mean = X.mean(axis=0)
        self._input_std = X.std(axis=0) + 1e-8

    # Apply the stored tanh normalisation (identity if not yet fitted).
    def _apply_input_norm(self, X: np.ndarray) -> np.ndarray:
        """Apply tanh normalisation using stored mean/std."""
        if self._input_mean is None:
            return X
        z = (X - self._input_mean) / self._input_std
        return np.tanh(z)

    # Full training routine: normalise → window → AdamW + cosine warm restarts + early stopping.
    def fit(
        self,
        X_train,
        y_train,
        sample_weight=None,
        X_val=None,
        y_val=None,
    ) -> None:
        # Seed every RNG so the same data + hyperparameters produce the same fitted weights.
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Tanh normalisation: fit on train only, apply to train and (if present) validation.
        X_train_arr = X_train.values if hasattr(X_train, "values") else X_train
        self._fit_input_norm(X_train_arr)
        X_train_normed = self._apply_input_norm(X_train_arr)

        # Build sliding-window sequences for both train and validation.
        import pandas as pd
        X_train_df = pd.DataFrame(X_train_normed, index=getattr(X_train, 'index', None))
        X_seq, y_seq, w_seq, _ = create_sequences(
            X_train_df, y_train, sample_weight, window=self.window
        )

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_arr = X_val.values if hasattr(X_val, "values") else X_val
            X_val_normed = self._apply_input_norm(X_val_arr)
            X_val_df = pd.DataFrame(X_val_normed, index=getattr(X_val, 'index', None))
            X_val_seq, y_val_seq, _, _ = create_sequences(
                X_val_df, y_val, window=self.window
            )

        # Move tensors onto the selected device once; default to ones if no weights supplied.
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

        # Mini-batch shuffling helps regularise the LSTM despite the short window.
        train_ds = TensorDataset(X_t, y_t, w_t)
        train_dl = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True)

        # Instantiate the underlying nn.Module on the selected device.
        self.net = LSTMClassifier(
            n_features=self.n_features, n_classes=self.n_classes
        ).to(self.device)

        # Class-frequency-inverse weighting handles label imbalance inside the loss.
        class_counts = np.bincount(y_seq, minlength=self.n_classes)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / class_weights.sum() * self.n_classes
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        # Per-sample loss reduction so AFML sample weights multiply correctly into each term.
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_t, reduction="none",
            label_smoothing=LSTM_LABEL_SMOOTHING,
        )
        criterion_val = nn.CrossEntropyLoss(
            weight=class_weights_t,
            label_smoothing=LSTM_LABEL_SMOOTHING,
        )
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=LSTM_LR, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=LSTM_WARMRESTART_T0,
            T_mult=LSTM_WARMRESTART_TMULT,
            eta_min=1e-5,
        )

        # Per-epoch convergence is logged at DEBUG; outer pipeline-level progress sits in pipeline.py.
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(LSTM_EPOCHS):
            self.net.train()
            epoch_loss = 0.0
            n_batches = 0

            # One mini-batch step: forward, weighted loss, backward, gradient clip, optimiser step.
            for X_b, y_b, w_b in train_dl:
                optimizer.zero_grad()
                logits = self.net(X_b)
                per_sample_loss = criterion(logits, y_b)
                weighted_loss = (per_sample_loss * w_b).mean()
                weighted_loss.backward()
                # Gradient clipping prevents the exploding-gradient pathology common in LSTMs.
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), LSTM_GRAD_CLIP_NORM
                )
                optimizer.step()

                epoch_loss += weighted_loss.item()
                n_batches += 1

            scheduler.step()
            avg_train_loss = epoch_loss / max(n_batches, 1)

            # Validation pass: track best weights and the patience counter for early stopping.
            if has_val:
                self.net.eval()
                with torch.no_grad():
                    val_logits = self.net(X_val_t)
                    val_loss = criterion_val(val_logits, y_val_t).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.net.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                logger.debug(
                    "LSTM epoch %d/%d: train=%.4f val=%.4f best=%.4f patience=%d/%d",
                    epoch + 1, LSTM_EPOCHS, avg_train_loss, val_loss,
                    best_val_loss, patience_counter, LSTM_PATIENCE,
                )

                if patience_counter >= LSTM_PATIENCE:
                    logger.debug(
                        "LSTM early stopping at epoch %d (best val loss: %.4f).",
                        epoch + 1, best_val_loss,
                    )
                    break
            else:
                logger.debug(
                    "LSTM epoch %d/%d: train=%.4f (no validation)",
                    epoch + 1, LSTM_EPOCHS, avg_train_loss,
                )

        # Restore the best validation weights (no-op if no validation was supplied).
        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.net.eval()
        logger.debug(
            "LSTM fitted: %d sequences, %d epochs completed, device=%s.",
            len(y_seq), epoch + 1, self.device,
        )

    # Class probabilities at inference; stores valid_indices so the caller can align predictions to dates.
    def predict_proba(self, X) -> np.ndarray:
        X_arr = X.values if hasattr(X, "values") else X
        X_normed = self._apply_input_norm(X_arr)
        import pandas as pd
        X_df = pd.DataFrame(X_normed, index=getattr(X, 'index', None))
        X_seq, valid_indices = create_sequences(X_df, window=self.window)
        self.last_valid_indices = valid_indices

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

        self.net.eval()
        with torch.no_grad():
            logits = self.net(X_t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()

        return proba

    # Hard-label prediction via argmax.
    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    # Raw pre-softmax logits for downstream calibration.
    def predict_logits(self, X) -> np.ndarray:
        """Return raw pre-softmax logits, shape ``(n_seq, n_classes)``."""
        X_arr = X.values if hasattr(X, "values") else X
        X_normed = self._apply_input_norm(X_arr)
        import pandas as pd
        X_df = pd.DataFrame(X_normed, index=getattr(X, 'index', None))
        X_seq, valid_indices = create_sequences(X_df, window=self.window)
        self.last_valid_indices = valid_indices

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

        self.net.eval()
        with torch.no_grad():
            logits = self.net(X_t).cpu().numpy()

        return logits

    def get_name(self) -> str:
        return "LSTM"