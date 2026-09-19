"""
A3: Predicting Car Price - Multinomial Logistic Regression, implemented from scratch.

This mirrors the style of the from-scratch LinearRegression built for A2 (same
Xavier/zeros initialization, batch/mini-batch/stochastic gradient descent, optional
momentum, early stopping on a validation set) but swaps the regression math for
multinomial (softmax) classification, and adds the classification metrics required
by A3 Task 1 (accuracy, per-class precision/recall/f1, macro- and support-weighted
averages) plus the optional Ridge (L2) penalty required by A3 Task 2.

This file is imported both by the notebook (for training/experimentation) and by
the deployed Dash app (app.py), so there is a single source of truth for the model
that gets pickled and later unpickled in production.
"""

import numpy as np
import pandas as pd


def add_intercept(X):
    """Prepend a column of ones so theta[0] acts as the bias term."""
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)


def one_hot(y, k):
    """Turn integer class labels 0..k-1 into a one-hot matrix of shape (m, k)."""
    y = np.asarray(y).astype(int)
    onehot = np.zeros((y.shape[0], k))
    onehot[np.arange(y.shape[0]), y] = 1
    return onehot


def softmax(z):
    """Row-wise softmax, shifted by the row max for numerical stability."""
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class LogisticRegression:
    """Multinomial (softmax) Logistic Regression trained with gradient descent.

    Parameters
    ----------
    k : int
        Number of classes.
    n : int
        Number of input features (not counting the bias term).
    method : {"batch", "mini", "sto"}
        Batch, mini-batch, or stochastic gradient descent.
    bs : int
        Batch size, only used when method="mini".
    alpha : float
        Learning rate.
    max_iter : int
        Maximum number of epochs.
    init_method : {"zeros", "xavier"}
        Weight initialization strategy. Xavier draws from
        U[-1/sqrt(n), 1/sqrt(n)], where n is the number of input features.
    use_momentum : bool
        Whether to accumulate a velocity term across updates.
    momentum : float
        Momentum coefficient in (0, 1), only used when use_momentum=True.
    regularization : {None, "ridge"}
        None for plain (unregularized) softmax regression, "ridge" to add an
        L2 penalty lambda_ * sum(theta_j ** 2) over all non-bias weights.
    lambda_ : float
        Ridge regularization strength.
    """

    def __init__(
        self,
        k,
        n,
        method="mini",
        bs=50,
        alpha=0.01,
        max_iter=1000,
        init_method="xavier",
        use_momentum=True,
        momentum=0.9,
        regularization=None,
        lambda_=0.0,
        verbose=False,
        random_state=42,
    ):
        self.k = k
        self.n = n
        self.method = method
        self.bs = bs
        self.alpha = alpha
        self.max_iter = max_iter
        self.init_method = init_method
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.regularization = regularization
        self.lambda_ = lambda_
        self.verbose = verbose
        self.random_state = random_state

        self.theta = None
        self.classes_ = np.arange(k)
        self.train_losses = []
        self.val_losses = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _init_theta(self):
        rng = np.random.RandomState(self.random_state)
        if self.init_method == "zeros":
            self.theta = np.zeros((self.n + 1, self.k))
        elif self.init_method == "xavier":
            lower, upper = -(1.0 / np.sqrt(self.n)), (1.0 / np.sqrt(self.n))
            self.theta = lower + rng.rand(self.n + 1, self.k) * (upper - lower)
        else:
            raise ValueError("init_method must be 'zeros' or 'xavier'")

    def h_theta(self, X):
        """X already has the intercept column; returns softmax probabilities."""
        return softmax(X @ self.theta)

    def _loss(self, X, Y_onehot):
        m = X.shape[0]
        h = np.clip(self.h_theta(X), 1e-12, 1 - 1e-12)
        loss = -np.sum(Y_onehot * np.log(h)) / m
        if self.regularization == "ridge":
            # bias term (row 0) is never regularized
            loss += self.lambda_ * np.sum(self.theta[1:, :] ** 2)
        return loss

    def _gradient(self, X, Y_onehot):
        m = X.shape[0]
        h = self.h_theta(X)
        grad = X.T @ (h - Y_onehot) / m
        if self.regularization == "ridge":
            reg_grad = 2 * self.lambda_ * self.theta
            reg_grad[0, :] = 0
            grad = grad + reg_grad
        return grad

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        rng = np.random.RandomState(self.random_state)

        X_train = add_intercept(np.asarray(X_train, dtype=float))
        Y_train_oh = one_hot(y_train, self.k)
        m = X_train.shape[0]

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_i = add_intercept(np.asarray(X_val, dtype=float))
            Y_val_oh = one_hot(y_val, self.k)

        self._init_theta()
        velocity = np.zeros_like(self.theta)
        self.train_losses, self.val_losses = [], []
        prev_val_loss = None

        for epoch in range(self.max_iter):
            if self.method == "batch":
                batches = [(X_train, Y_train_oh)]
            elif self.method == "sto":
                idx = rng.permutation(m)
                batches = [(X_train[i:i + 1], Y_train_oh[i:i + 1]) for i in idx]
            elif self.method == "mini":
                idx = rng.permutation(m)
                X_shuf, Y_shuf = X_train[idx], Y_train_oh[idx]
                batches = [
                    (X_shuf[i:i + self.bs], Y_shuf[i:i + self.bs])
                    for i in range(0, m, self.bs)
                ]
            else:
                raise ValueError("method must be 'batch', 'mini', or 'sto'")

            for X_b, Y_b in batches:
                grad = self._gradient(X_b, Y_b)
                if self.use_momentum:
                    velocity = self.momentum * velocity - self.alpha * grad
                    self.theta = self.theta + velocity
                else:
                    self.theta = self.theta - self.alpha * grad

            train_loss = self._loss(X_train, Y_train_oh)
            self.train_losses.append(train_loss)

            if has_val:
                val_loss = self._loss(X_val_i, Y_val_oh)
                self.val_losses.append(val_loss)
                if prev_val_loss is not None and np.allclose(val_loss, prev_val_loss, atol=1e-7):
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
                prev_val_loss = val_loss

            if self.verbose and epoch % max(1, self.max_iter // 10) == 0:
                msg = f"Epoch {epoch}: train_loss={train_loss:.5f}"
                if has_val:
                    msg += f", val_loss={val_loss:.5f}"
                print(msg)

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X):
        X = add_intercept(np.asarray(X, dtype=float))
        return self.h_theta(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    # ------------------------------------------------------------------
    # Classification metrics, implemented from scratch (A3 Task 1)
    # ------------------------------------------------------------------

    @staticmethod
    def accuracy(y_true, y_pred):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        return float(np.sum(y_true == y_pred) / len(y_true))

    @staticmethod
    def _confusion_counts(y_true, y_pred, c):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        tn = np.sum((y_true != c) & (y_pred != c))
        return tp, fp, fn, tn

    @classmethod
    def precision(cls, y_true, y_pred, c):
        tp, fp, _, _ = cls._confusion_counts(y_true, y_pred, c)
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    @classmethod
    def recall(cls, y_true, y_pred, c):
        tp, _, fn, _ = cls._confusion_counts(y_true, y_pred, c)
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    @classmethod
    def f1_score(cls, y_true, y_pred, c):
        p, r = cls.precision(y_true, y_pred, c), cls.recall(y_true, y_pred, c)
        return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    @classmethod
    def macro_precision(cls, y_true, y_pred, classes):
        return float(np.mean([cls.precision(y_true, y_pred, c) for c in classes]))

    @classmethod
    def macro_recall(cls, y_true, y_pred, classes):
        return float(np.mean([cls.recall(y_true, y_pred, c) for c in classes]))

    @classmethod
    def macro_f1(cls, y_true, y_pred, classes):
        return float(np.mean([cls.f1_score(y_true, y_pred, c) for c in classes]))

    @staticmethod
    def _supports(y_true, classes):
        y_true = np.asarray(y_true)
        supports = np.array([np.sum(y_true == c) for c in classes], dtype=float)
        return supports / supports.sum()

    @classmethod
    def weighted_precision(cls, y_true, y_pred, classes):
        weights = cls._supports(y_true, classes)
        values = np.array([cls.precision(y_true, y_pred, c) for c in classes])
        return float(np.sum(weights * values))

    @classmethod
    def weighted_recall(cls, y_true, y_pred, classes):
        weights = cls._supports(y_true, classes)
        values = np.array([cls.recall(y_true, y_pred, c) for c in classes])
        return float(np.sum(weights * values))

    @classmethod
    def weighted_f1(cls, y_true, y_pred, classes):
        weights = cls._supports(y_true, classes)
        values = np.array([cls.f1_score(y_true, y_pred, c) for c in classes])
        return float(np.sum(weights * values))

    @classmethod
    def classification_report(cls, y_true, y_pred, target_names=None):
        """From-scratch equivalent of sklearn.metrics.classification_report.

        Returns (report_df, accuracy) where report_df has one row per class
        plus 'macro avg' and 'weighted avg' rows, matching sklearn's layout.
        """
        y_true = np.asarray(y_true)
        classes = np.unique(np.concatenate([y_true, np.asarray(y_pred)])).astype(int)
        classes.sort()
        if target_names is None:
            target_names = [str(c) for c in classes]

        rows = []
        for c, name in zip(classes, target_names):
            rows.append({
                "class": name,
                "precision": cls.precision(y_true, y_pred, c),
                "recall": cls.recall(y_true, y_pred, c),
                "f1-score": cls.f1_score(y_true, y_pred, c),
                "support": int(np.sum(y_true == c)),
            })

        rows.append({
            "class": "macro avg",
            "precision": cls.macro_precision(y_true, y_pred, classes),
            "recall": cls.macro_recall(y_true, y_pred, classes),
            "f1-score": cls.macro_f1(y_true, y_pred, classes),
            "support": len(y_true),
        })
        rows.append({
            "class": "weighted avg",
            "precision": cls.weighted_precision(y_true, y_pred, classes),
            "recall": cls.weighted_recall(y_true, y_pred, classes),
            "f1-score": cls.weighted_f1(y_true, y_pred, classes),
            "support": len(y_true),
        })

        report_df = pd.DataFrame(rows).set_index("class")
        return report_df, cls.accuracy(y_true, y_pred)

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    def plot_feature_importance(self, feature_names, top_n=20):
        """Bar chart of mean |coefficient| across classes (input must be scaled)."""
        import matplotlib.pyplot as plt

        coefs = self.theta[1:, :]  # drop the bias row
        importance = np.mean(np.abs(coefs), axis=1)
        order = np.argsort(importance)[-top_n:]

        fig, ax = plt.subplots(figsize=(8, max(4, len(order) * 0.3)))
        ax.barh(np.array(feature_names)[order], importance[order])
        ax.set_xlabel("Mean |coefficient| across classes")
        ax.set_title("Feature Importance — Multinomial Logistic Regression")
        fig.tight_layout()
        return fig
