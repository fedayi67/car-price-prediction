# 🚗 Car Price Prediction III — Classification (Assignment 3)

## Project Overview

This project extends the Car Price Prediction series (A1: sklearn baseline,
A2: from-scratch Linear Regression + MLflow + Docker) by **reframing the
problem as 4-class classification**. `selling_price` is bucketed into 4
price classes (`0` = Budget … `3` = Luxury), and a **multinomial Logistic
Regression is implemented entirely from scratch**, including every metric in
scikit-learn's `classification_report` (accuracy, per-class precision/recall/f1,
macro- and support-weighted averages) and an optional Ridge (L2) penalty.

The project covers:

- Data cleaning (reused, unchanged, from A1/A2)
- Bucketing `selling_price` into 4 classes with `pd.cut()`
- Feature preprocessing (reused from A2)
- A from-scratch `LogisticRegression` class (Xavier/zeros init, batch/mini-batch/
  stochastic gradient descent, momentum, optional Ridge)
- From-scratch classification metrics, verified against `sklearn.metrics.classification_report`
- MLflow experiment tracking and Model Registry (staged)
- A Dash web application, Dockerized, with a GitHub Actions CI/CD pipeline
  (unit tests → build/push → deploy)

---

## 📂 Dataset

```text
Cars.csv
```

Same dataset and cleaning rules as A1/A2 (see the notebook for full detail):
owner mapped to `1..4` (Test Drive Cars removed), CNG/LPG rows dropped,
`mileage`/`engine`/`max_power` parsed out of their unit strings, `brand`
extracted from `name`, `torque` dropped, rare brands (<10 occurrences) grouped
into `Rare Brands`, and `car_age` engineered from `year` (with `year` itself
dropped to avoid the perfect multicollinearity noted in A1/A2).

## 🎯 Target: `price_class`

`selling_price` is right-skewed, so bucketing it directly with equal-width
`pd.cut()` puts >99% of vehicles in a single bin. Instead, `pd.cut()` is
applied to `log(selling_price)` (the same transform A1/A2 used for
regression), and the resulting bin edges are converted back to the original
price scale for readability. This yields a realistic, moderately imbalanced
4-class split (~6% / 53% / 37% / 4%) — deliberately kept imbalanced, since
that is exactly the scenario the assignment uses to motivate `weighted`
averaging over `macro` averaging.

## 🧠 Task 1 — Logistic Regression From Scratch

`app/code/model.py`'s `LogisticRegression` class implements multinomial
(softmax) logistic regression trained with gradient descent, plus every
metric required by Task 1, computed with no scikit-learn:

- `accuracy(y_true, y_pred)`
- `precision`, `recall`, `f1_score` (per class)
- `macro_precision`, `macro_recall`, `macro_f1`
- `weighted_precision`, `weighted_recall`, `weighted_f1`
- `classification_report(y_true, y_pred)` — lays these out the same way as
  `sklearn.metrics.classification_report`, including the `support` column

The notebook verifies these against `sklearn.metrics.classification_report`
both on mock data and on the real test-set predictions; they match to several
decimal places.

**What does `support` mean?** It is the number of true (ground-truth) examples
of that class in the evaluation set — it does not depend on the model's
predictions at all. It is what `weighted avg` weights by, which is why
`weighted avg` and `macro avg` diverge noticeably on an imbalanced dataset
like this one.

## 🧮 Task 2 — Ridge (L2) Logistic Regression

The same class accepts `regularization="ridge"` and a `lambda_` strength. The
intercept (`theta[0]`) is excluded from the penalty in both the loss and the
gradient. The notebook compares several `lambda_` values against the
unregularized model.

## 🧪 Task 3 — MLflow, Model Registry, and CI/CD

### MLflow experiment tracking

The notebook points `mlflow.set_tracking_uri()` at the CSIM server
(`http://mlflow.ml.brain.cs.ait.ac.th/`) first, with experiment name
`<student_id>-a3`. **If that server is unreachable** (e.g. running outside the
AIT/CSIM network or VPN), it automatically falls back to a local
SQLite-backed store (`sqlite:///mlflow.db`) so the full experiment + Model
Registry workflow can still be exercised end-to-end. Re-running the notebook
on the CSIM network logs to the real server with no code changes needed.

The dataset itself is never logged (only params, metrics, and the model
artifact), per the assignment's instructions.

### Model Registry

The best run (by test-set `weighted_f1`) is registered as
`<student_id>-a3-model` and moved to the **Staging** stage automatically. When
run against the real CSIM server, take screenshots of the Experiments and
Models tabs there for the submission — this repo's notebook, when run
locally, uses the SQLite fallback and therefore does not have those
screenshots.

### CI/CD (`.github/workflows/ci-cd.yml`)

1. **`test` job** — on every push, installs `app/code/requirements.txt` +
   `pytest`, then runs `app/code/tests/test_model.py`, which contains the two
   required unit tests: (1) the model accepts the expected input shape, and
   (2) `predict`/`predict_proba` return the expected output shape.
2. **`deploy` job** — runs only if `test` passes (`needs: test`). Builds the
   Docker image from `app/` and pushes it to Docker Hub. Requires
   `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` repository secrets.
   A commented-out SSH step is included to redeploy the container on
   `ml.brain.cs.ait.ac.th` via `docker compose pull && docker compose up -d`;
   it needs `SSH_HOST` / `SSH_USER` / `SSH_PRIVATE_KEY` secrets before it can
   run (the private half of the key pair the TA installs on the server, per
   the assignment PDF).

## 🖥️ Deployment — Dash Web Application

```text
app/
├── Dockerfile
├── docker-compose.yaml
└── code/
    ├── app.py                       # Dash app: predicts a price class + range
    ├── model.py                     # the LogisticRegression class (single source of truth)
    ├── car_price_classifier_a3.pkl  # serialized {preprocessor, model, price bins}
    ├── requirements.txt
    └── tests/
        └── test_model.py
```

The app collects the same vehicle fields as A1/A2, applies the identical
preprocessing pipeline, and shows the predicted price **class** (Budget /
Economy / Premium / Luxury), the corresponding price range in the selected
currency, and the model's class probabilities.

Run locally:

```bash
cd app
docker compose up --build
# open http://localhost:8060
```

To deploy on the CSIM server, uncomment the Traefik labels block in
`app/docker-compose.yaml` (subdomain `web-st127302-a3` by default — change it
to your own), then follow the same `ml.brain.cs.ait.ac.th` deployment steps
used in A2 (SSH in with your registered private key, `docker compose up -d`).

## 📓 Notebook

`Car Price Prediction III.ipynb` walks through the full pipeline end-to-end —
cleaning → bucketing → preprocessing → from-scratch Logistic Regression →
metric verification → Ridge comparison → MLflow experiments/registry →
final model serialization → feature importance → discussion — and was
executed top-to-bottom to produce the outputs it contains.

## Github links:

_add your repository URL(s) here before submitting_
