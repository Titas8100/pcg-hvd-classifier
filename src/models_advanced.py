"""
Advanced Machine Learning & Deep Learning classification models.
Provides:
1. SOTA ML Classifiers: Random Forest, Extra Trees, Gradient Boosting, Tuned RBF SVM, MLP.
2. PyTorch 1D-CNN Architecture tailored for PCG time-series waveforms.
3. Stratified 10-Fold and 5-Fold Cross-Validation evaluation.
4. Comprehensive multi-class metrics: Accuracy, Precision, Recall, Macro-F1, Weighted-F1, Confusion Matrices.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any, Optional

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def get_advanced_ml_models() -> Dict[str, Any]:
    """
    Returns a dictionary of competitive ML classifiers wrapped with StandardScaler in sklearn Pipelines.
    """
    return {
        "Quadratic SVM (Baseline)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel='poly', degree=2, C=1.0, gamma='scale', random_state=42))
        ]),
        "Tuned RBF SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42))
        ]),
        "KNN (k=5, Distance-Weighted)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean'))
        ]),
        "Random Forest (300 Trees)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=3, random_state=42))
        ]),
        "Extra Trees Classifier": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(n_estimators=300, max_depth=15, random_state=42))
        ]),
        "Gradient Boosting (GBDT)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=4, random_state=42))
        ]),
        "Multi-Layer Perceptron (MLP)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600, activation='relu', random_state=42))
        ]),
    }


def evaluate_models_cross_validation(
    models: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Evaluates each model using Stratified K-Fold Cross-Validation.
    Returns:
        results_df: DataFrame with mean and std for Accuracy, Macro-F1, Weighted-F1, Precision, Recall.
        confusion_matrices: Dict of out-of-fold confusion matrices for each model.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    summary_rows = []
    confusion_matrices = {}

    num_classes = len(np.unique(y))

    for name, model in models.items():
        print(f"  Evaluating {name} ({n_splits}-fold CV)...")
        fold_accs = []
        fold_macro_f1s = []
        fold_weighted_f1s = []
        fold_precisions = []
        fold_recalls = []
        
        oof_preds = np.zeros(len(y), dtype=int)

        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            oof_preds[test_idx] = y_pred

            fold_accs.append(accuracy_score(y_te, y_pred) * 100.0)
            fold_macro_f1s.append(f1_score(y_te, y_pred, average='macro', zero_division=0) * 100.0)
            fold_weighted_f1s.append(f1_score(y_te, y_pred, average='weighted', zero_division=0) * 100.0)
            fold_precisions.append(precision_score(y_te, y_pred, average='macro', zero_division=0) * 100.0)
            fold_recalls.append(recall_score(y_te, y_pred, average='macro', zero_division=0) * 100.0)

        cm = confusion_matrix(y, oof_preds)
        confusion_matrices[name] = cm

        summary_rows.append({
            "Model": name,
            "Accuracy% (Mean ± Std)": f"{np.mean(fold_accs):.2f} ± {np.std(fold_accs):.2f}",
            "Accuracy_Mean": round(np.mean(fold_accs), 2),
            "Accuracy_Std": round(np.std(fold_accs), 2),
            "Macro_F1%": round(np.mean(fold_macro_f1s), 2),
            "Weighted_F1%": round(np.mean(fold_weighted_f1s), 2),
            "Precision%": round(np.mean(fold_precisions), 2),
            "Recall%": round(np.mean(fold_recalls), 2),
        })

    results_df = pd.DataFrame(summary_rows).sort_values(by="Accuracy_Mean", ascending=False).reset_index(drop=True)
    return results_df, confusion_matrices


# =========================================================================
# Deep Learning: 1D-CNN Architecture for Raw PCG Waveforms
# =========================================================================

class PCG1DCNN(nn.Module):
    """
    1D Convolutional Neural Network for end-to-end heart sound PCG classification.
    Processes normalized 1D PCG time series waveforms directly.
    """
    def __init__(self, num_classes: int = 5):
        super(PCG1DCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 4
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        out = self.classifier(feat)
        return out


def train_pcg_1d_cnn(
    signals: List[np.ndarray],
    labels_idx: np.ndarray,
    target_length: int = 16000,
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 1e-3,
    test_size: float = 0.30,
    random_state: int = 42
) -> Tuple[float, np.ndarray, Dict[str, List[float]]]:
    """
    Trains and evaluates the 1D-CNN on fixed-length normalized PCG waveforms.
    Returns:
        test_acc: Accuracy on test set.
        cm: Confusion matrix.
        history: Training/validation loss and accuracy curves.
    """
    from sklearn.model_selection import train_test_split
    
    # 1. Pad or trim signals to target_length
    processed_signals = []
    for sig in signals:
        if len(sig) >= target_length:
            processed_signals.append(sig[:target_length])
        else:
            pad = target_length - len(sig)
            processed_signals.append(np.pad(sig, (0, pad), mode='constant'))
            
    X_arr = np.array(processed_signals, dtype=np.float32)[:, np.newaxis, :]  # (N, 1, L)
    y_arr = labels_idx.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=test_size, random_state=random_state, stratify=y_arr
    )

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PCG1DCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "test_acc": []}

    print(f"  Training 1D-CNN on {device} ({epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch_y)
            
        scheduler.step()
        epoch_loss = running_loss / len(train_dataset)
        history["train_loss"].append(epoch_loss)

        # Eval
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
        acc = (correct / len(test_dataset)) * 100.0
        history["test_acc"].append(acc)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"    Epoch {epoch:02d}/{epochs} - Loss: {epoch_loss:.4f} - Test Acc: {acc:.2f}%")

    # Final evaluation & confusion matrix
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            preds = torch.argmax(model(batch_x), dim=1)
            all_preds.extend(preds.cpu().numpy())

    all_preds = np.array(all_preds)
    final_acc = accuracy_score(y_test, all_preds) * 100.0
    cm = confusion_matrix(y_test, all_preds)

    return final_acc, cm, history
