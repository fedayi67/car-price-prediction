"""
Unit tests for the from-scratch LogisticRegression model (A3 Task 3, Objective 3).

Only two things are required by the assignment: (1) the model accepts the input
shape it expects, and (2) its output has the expected shape. Both are tested
here against a freshly-trained tiny model, so these tests do not depend on the
serialized car_price_classifier_a3.pkl artifact existing on disk (that keeps
the GitHub Actions test job fast and independent of the notebook having been run).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import LogisticRegression  # noqa: E402


K_CLASSES = 4
N_FEATURES = 6
N_SAMPLES = 50


@pytest.fixture
def trained_model():
    rng = np.random.RandomState(0)
    X = rng.randn(N_SAMPLES, N_FEATURES)
    y = rng.randint(0, K_CLASSES, size=N_SAMPLES)

    model = LogisticRegression(
        k=K_CLASSES,
        n=N_FEATURES,
        method="batch",
        alpha=0.1,
        max_iter=20,
        init_method="xavier",
        use_momentum=True,
        momentum=0.9,
    )
    model.fit(X, y)
    return model


def test_model_accepts_expected_input(trained_model):
    """The model should accept a 2D array with N_FEATURES columns and produce
    a prediction for every row, without raising."""
    rng = np.random.RandomState(1)
    X_new = rng.randn(10, N_FEATURES)

    predictions = trained_model.predict(X_new)

    assert predictions is not None
    assert len(predictions) == 10


def test_output_shape(trained_model):
    """predict() should return one class label per input row, each a valid
    class index; predict_proba() should return a (n_samples, k) probability
    matrix whose rows sum to 1."""
    rng = np.random.RandomState(2)
    n_query = 7
    X_new = rng.randn(n_query, N_FEATURES)

    predictions = trained_model.predict(X_new)
    probabilities = trained_model.predict_proba(X_new)

    assert predictions.shape == (n_query,)
    assert probabilities.shape == (n_query, K_CLASSES)
    assert np.all((predictions >= 0) & (predictions < K_CLASSES))
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(n_query), rtol=1e-6)
