"""
Exact Reproduction of Report Experiments (Tables 1a, 1b, 2a, 2b).
Faithfully mirrors the experimental procedures described in Sections 5 and 6 of Titas Routh's report:
- 70% Train, 30% Test split (stratified for consistency).
- Feature dropout experiments (with and without normalization).
- Feature selection percentage experiments (10% to 100%).
- Quadratic SVM (Polynomial kernel degree 2) and KNN classifiers.
- Generates side-by-side comparison tables between original Report results and Python reproductions.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from typing import Dict, Tuple, List, Optional


# Ground Truth results reported in Titas Routh's report
REPORT_TABLE_1A = [
    {"Classifier": "SVM", "Dropout%": 82, "Train%": 70, "Test%": 30, "Selection%": 100, "Normalized": "N", "Report_Accuracy%": 41.80},
    {"Classifier": "SVM", "Dropout%": 70, "Train%": 70, "Test%": 30, "Selection%": 100, "Normalized": "N", "Report_Accuracy%": 21.64},
    {"Classifier": "SVM", "Dropout%": 47, "Train%": 70, "Test%": 30, "Selection%": 100, "Normalized": "N", "Report_Accuracy%": 59.00},
    {"Classifier": "SVM", "Dropout%": 47, "Train%": 70, "Test%": 30, "Selection%": 100, "Normalized": "Y", "Report_Accuracy%": 50.00},
    {"Classifier": "SVM", "Dropout%": 0,  "Train%": 70, "Test%": 30, "Selection%": 100, "Normalized": "N", "Report_Accuracy%": 69.16},
    {"Classifier": "SVM", "Dropout%": 0,  "Train%": 70, "Test%": 30, "Selection%": 100, "Normalized": "Y", "Report_Accuracy%": 44.50},
]

REPORT_TABLE_1B = [
    {"Selection%": 10,  "Report_SVM_Acc%": 88.61, "Report_KNN_Acc%": 92.96},
    {"Selection%": 20,  "Report_SVM_Acc%": 83.28, "Report_KNN_Acc%": 82.87},
    {"Selection%": 30,  "Report_SVM_Acc%": 83.16, "Report_KNN_Acc%": 79.21},
    {"Selection%": 40,  "Report_SVM_Acc%": 94.49, "Report_KNN_Acc%": 95.63},
    {"Selection%": 50,  "Report_SVM_Acc%": 91.51, "Report_KNN_Acc%": 92.55},
    {"Selection%": 60,  "Report_SVM_Acc%": 95.39, "Report_KNN_Acc%": 91.85},
    {"Selection%": 70,  "Report_SVM_Acc%": 88.86, "Report_KNN_Acc%": 84.64},
    {"Selection%": 80,  "Report_SVM_Acc%": 80.03, "Report_KNN_Acc%": 75.33},
    {"Selection%": 90,  "Report_SVM_Acc%": 76.45, "Report_KNN_Acc%": 73.95},
    {"Selection%": 100, "Report_SVM_Acc%": 81.13, "Report_KNN_Acc%": 76.81},
]

REPORT_TABLE_2A = [
    {"Classifier": "SVM", "Dropout%": 96, "Train%": 70, "Test%": 30, "Selection%": 100, "Normalized": "N", "Report_Accuracy%": 21.14},
    {"Classifier": "SVM", "Dropout%": 96, "Train%": 70, "Test%": 30, "Selection%": 100, "Normalized": "Y", "Report_Accuracy%": 20.00},
    {"Classifier": "SVM", "Dropout%": 94, "Train%": 70, "Test%": 30, "Selection%": 100, "Normalized": "N", "Report_Accuracy%": 53.12},
    {"Classifier": "SVM", "Dropout%": 94, "Train%": 70, "Test%": 30, "Selection%": 100, "Normalized": "Y", "Report_Accuracy%": 52.86},
    {"Classifier": "SVM", "Dropout%": 87, "Train%": 70, "Test%": 30, "Selection%": 100, "Normalized": "N", "Report_Accuracy%": 56.46},
    {"Classifier": "SVM", "Dropout%": 83, "Train%": 70, "Test%": 30, "Selection%": 100, "Normalized": "N", "Report_Accuracy%": 51.82},
]

REPORT_TABLE_2B = [
    {"Selection%": 10,  "Report_KNN_Acc%": 79.96},
    {"Selection%": 20,  "Report_KNN_Acc%": 82.66},
    {"Selection%": 30,  "Report_KNN_Acc%": 85.47},
    {"Selection%": 40,  "Report_KNN_Acc%": 90.66},
    {"Selection%": 50,  "Report_KNN_Acc%": 79.59},
    {"Selection%": 60,  "Report_KNN_Acc%": 82.09},
    {"Selection%": 70,  "Report_KNN_Acc%": 84.10},
    {"Selection%": 80,  "Report_KNN_Acc%": 78.10},
    {"Selection%": 90,  "Report_KNN_Acc%": 71.30},
    {"Selection%": 100, "Report_KNN_Acc%": 73.70},
]


def create_base_classifiers():
    """
    Returns the exact base classifiers specified in report Section 5:
    - Quadratic SVM: SVC(kernel='poly', degree=2, C=1.0)
    - KNN: KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    """
    svm = SVC(kernel='poly', degree=2, C=1.0, gamma='scale', random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    return svm, knn


def apply_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    percentile: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies ANOVA F-value feature selection according to the specified percentile.
    Ensures that when percentile=100, all features are passed through.
    """
    if percentile >= 100:
        return X_train, X_test
    selector = SelectPercentile(score_func=f_classif, percentile=percentile)
    X_tr = selector.fit_transform(X_train, y_train)
    X_te = selector.transform(X_test)
    # Ensure at least 1 feature
    if X_tr.shape[1] == 0:
        X_tr = X_train[:, :1]
        X_te = X_test[:, :1]
    return X_tr, X_te


def run_table_1a(X_dwt: np.ndarray, y: np.ndarray, random_state: int = 42) -> pd.DataFrame:
    """
    Reproduces Table 1(a): Variation of accuracy % with feature dropout % for DWT features using SVM.
    Dropout %: 82%, 70%, 47%, 0% (equivalent to keeping (100 - dropout)%).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_dwt, y, test_size=0.30, random_state=random_state, stratify=y
    )

    results = []
    for row in REPORT_TABLE_1A:
        dropout = row["Dropout%"]
        norm = row["Normalized"]
        report_acc = row["Report_Accuracy%"]

        # 1. Normalization
        if norm == "Y":
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)
        else:
            X_tr = X_train.copy()
            X_te = X_test.copy()

        # 2. Dropout: keep (100 - dropout)%
        keep_pct = max(1, 100 - dropout)
        X_tr_sel, X_te_sel = apply_feature_selection(X_tr, y_train, X_te, percentile=keep_pct)

        # 3. Fit Quadratic SVM
        clf = SVC(kernel='poly', degree=2, C=1.0, gamma='scale', random_state=random_state)
        clf.fit(X_tr_sel, y_train)
        preds = clf.predict(X_te_sel)
        py_acc = accuracy_score(y_test, preds) * 100.0

        results.append({
            "Classifier": "SVM",
            "Dropout%": dropout,
            "Kept_Features": X_tr_sel.shape[1],
            "Train%": 70,
            "Test%": 30,
            "Normalized": norm,
            "Report_Accuracy%": report_acc,
            "Python_Accuracy%": round(py_acc, 2),
            "Difference%": round(py_acc - report_acc, 2)
        })

    return pd.DataFrame(results)


def run_table_1b(X_dwt: np.ndarray, y: np.ndarray, random_state: int = 42) -> pd.DataFrame:
    """
    Reproduces Table 1(b): Variation of accuracy % with feature selection % for DWT features using SVM & KNN.
    Dropout = 0%, No Normalization, 70:30 Train:Test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_dwt, y, test_size=0.30, random_state=random_state, stratify=y
    )

    results = []
    for row in REPORT_TABLE_1B:
        sel_pct = row["Selection%"]
        rep_svm = row["Report_SVM_Acc%"]
        rep_knn = row["Report_KNN_Acc%"]

        # Feature selection
        X_tr_sel, X_te_sel = apply_feature_selection(X_train, y_train, X_test, percentile=sel_pct)

        # 1. Quadratic SVM
        svm = SVC(kernel='poly', degree=2, C=1.0, gamma='scale', random_state=random_state)
        svm.fit(X_tr_sel, y_train)
        svm_acc = accuracy_score(y_test, svm.predict(X_te_sel)) * 100.0

        # 2. KNN
        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        knn.fit(X_tr_sel, y_train)
        knn_acc = accuracy_score(y_test, knn.predict(X_te_sel)) * 100.0

        results.append({
            "Selection%": sel_pct,
            "Num_Features": X_tr_sel.shape[1],
            "Report_SVM_Acc%": rep_svm,
            "Python_SVM_Acc%": round(svm_acc, 2),
            "Report_KNN_Acc%": rep_knn,
            "Python_KNN_Acc%": round(knn_acc, 2),
        })

    return pd.DataFrame(results)


def run_table_2a(X_mfcc: np.ndarray, y: np.ndarray, random_state: int = 42) -> pd.DataFrame:
    """
    Reproduces Table 2(a): Variation of accuracy % with feature dropout % for MFCC features using SVM.
    Dropout %: 96%, 94%, 87%, 83%.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_mfcc, y, test_size=0.30, random_state=random_state, stratify=y
    )

    results = []
    for row in REPORT_TABLE_2A:
        dropout = row["Dropout%"]
        norm = row["Normalized"]
        report_acc = row["Report_Accuracy%"]

        if norm == "Y":
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)
        else:
            X_tr = X_train.copy()
            X_te = X_test.copy()

        keep_pct = max(1, 100 - dropout)
        X_tr_sel, X_te_sel = apply_feature_selection(X_tr, y_train, X_te, percentile=keep_pct)

        clf = SVC(kernel='poly', degree=2, C=1.0, gamma='scale', random_state=random_state)
        clf.fit(X_tr_sel, y_train)
        py_acc = accuracy_score(y_test, clf.predict(X_te_sel)) * 100.0

        results.append({
            "Classifier": "SVM",
            "Dropout%": dropout,
            "Kept_Features": X_tr_sel.shape[1],
            "Train%": 70,
            "Test%": 30,
            "Normalized": norm,
            "Report_Accuracy%": report_acc,
            "Python_Accuracy%": round(py_acc, 2),
            "Difference%": round(py_acc - report_acc, 2)
        })

    return pd.DataFrame(results)


def run_table_2b(X_mfcc: np.ndarray, y: np.ndarray, random_state: int = 42) -> pd.DataFrame:
    """
    Reproduces Table 2(b): Variation of accuracy % with feature selection % for MFCC features using KNN.
    Dropout = 83% (keeping 17% base), Selection %: 10% to 100%, 70:30 Train:Test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_mfcc, y, test_size=0.30, random_state=random_state, stratify=y
    )

    # Initial dropout of 83% (or base MFCC set)
    X_tr_base, X_te_base = apply_feature_selection(X_train, y_train, X_test, percentile=max(1, 100 - 83))

    results = []
    for row in REPORT_TABLE_2B:
        sel_pct = row["Selection%"]
        rep_knn = row["Report_KNN_Acc%"]

        # Feature selection
        X_tr_sel, X_te_sel = apply_feature_selection(X_train, y_train, X_test, percentile=sel_pct)

        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        knn.fit(X_tr_sel, y_train)
        knn_acc = accuracy_score(y_test, knn.predict(X_te_sel)) * 100.0

        results.append({
            "Selection%": sel_pct,
            "Num_Features": X_tr_sel.shape[1],
            "Report_KNN_Acc%": rep_knn,
            "Python_KNN_Acc%": round(knn_acc, 2),
            "Difference%": round(knn_acc - rep_knn, 2)
        })

    return pd.DataFrame(results)


def run_fused_baseline(
    X_dwt: np.ndarray,
    X_mfcc: np.ndarray,
    y: np.ndarray,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluates combined DWT (24) + MFCC (19) features (43 features)
    with Quadratic SVM and KNN as proposed in the report.
    """
    X_fused = np.hstack([X_dwt, X_mfcc])
    X_train, X_test, y_train, y_test = train_test_split(
        X_fused, y, test_size=0.30, random_state=random_state, stratify=y
    )

    svm = SVC(kernel='poly', degree=2, C=1.0, gamma='scale', random_state=random_state)
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test)) * 100.0

    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_train, y_train)
    knn_acc = accuracy_score(y_test, knn.predict(X_test)) * 100.0

    return pd.DataFrame([
        {"Method": "Combined (DWT+MFCC)", "Classifier": "Quadratic SVM", "Accuracy%": round(svm_acc, 2)},
        {"Method": "Combined (DWT+MFCC)", "Classifier": "KNN (k=5)", "Accuracy%": round(knn_acc, 2)}
    ])
