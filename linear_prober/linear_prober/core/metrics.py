"""Scoring functions for the linear-probing downstream tasks.

Three task families are evaluated with a single held-out metric each:

  - binary classification      -> ROC-AUC on the positive-class probability
  - multiclass classification  -> ROC-AUC one-vs-rest, weighted average
  - regression                 -> coefficient of determination (R2)

Each family exposes two scorers: one for probabilistic estimators exposing
``predict_proba`` (LogisticRegression) and one for margin-based estimators
exposing only ``decision_function`` (RidgeClassifier, used in the
high-dimensional ``flatten`` regime where ``D >> N``).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score


def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax, numerically stabilised by max-subtraction."""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# =============================================================================
# Binary classification
# =============================================================================


def binary_auc_proba(model, X: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC from ``predict_proba`` positive-class column."""
    return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))


def binary_auc_decision(model, X: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC from a signed ``decision_function`` score."""
    return float(roc_auc_score(y, model.decision_function(X)))


# =============================================================================
# Multiclass classification (one-vs-rest, weighted)
# =============================================================================


def multiclass_auc_proba(model, X: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC OvR weighted from ``predict_proba``."""
    return float(
        roc_auc_score(y, model.predict_proba(X), multi_class="ovr", average="weighted")
    )


def multiclass_auc_decision(model, X: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC OvR weighted from ``decision_function`` via softmax.

    ``RidgeClassifier`` has no ``predict_proba``; its per-class margins are
    mapped to a probability simplex with a softmax before scoring.
    """
    return float(
        roc_auc_score(
            y,
            softmax(model.decision_function(X)),
            multi_class="ovr",
            average="weighted",
        )
    )


# =============================================================================
# Regression
# =============================================================================


def regression_r2(model, X: np.ndarray, y: np.ndarray) -> float:
    """Coefficient of determination on a single target dimension."""
    return float(r2_score(y, model.predict(X)))
