# Bank Marketing — Term Deposit Subscription Prediction

End-to-end binary classification project on the UCI Bank Marketing dataset: predicting which clients will subscribe to a term deposit after a direct marketing campaign.

This repository contains the **modelling work** (EDA, feature engineering, model selection, tuning, interpretation). The trained model is served through two separate deployment repos:

| Component | Repo | Live |
|---|---|---|
| Analysis & modelling (this repo) | [`bankmarketing_jupyter`](https://github.com/Itaru2018/bankmarketing_jupyter) | — |
| Prediction API (FastAPI):Back end| [`bank_marketing_fastapi`](https://github.com/Itaru2018/bank_marketing_fastapi) | https://bankmarketingstreamlit-production.up.railway.app/ |
| Web app (Streamlit):Frond end| [`bank_marketing_streamlit`](https://github.com/Itaru2018/bank_marketing_streamlit) | https://bankmarketingstreamlit-production.up.railway.app/ |

---

## Business framing

A bank runs telemarketing campaigns to sell term deposits. Most calls end in "no" — the dataset is heavily imbalanced (roughly 11% positives). 
The goal is twofold: **predict whether a given client will subscribe**, and **identify which features have the most impact on subscription success**, for further analysis of what drives a campaign to work.

Because of the imbalance and the asymmetric cost of a missed subscriber vs. a wasted call, the model is evaluated on **precision–recall trade-offs**, not accuracy. Accuracy is uninformative here: a model that predicts "no" for everyone already scores ~89%.

---

## Data

- Source: [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- ~41k rows, 20 features (client demographics, contact history, macroeconomic indicators)
- Target: `y` — did the client subscribe to a term deposit?

**Leakage note:** `duration` (call length in seconds) is known only *after* the call ends, so it cannot be used in a model intended to decide who to call. It is excluded from the modelling features.

---

## Approach

1. **EDA** — distributions, missingness (`unknown` categories), class imbalance and dimensionality reduction(PCA/LDA/UMAP) to check class separability.
2. **Preprocessing pipeline** — all steps wrapped in a scikit-learn `Pipeline` so that fitting happens inside cross-validation folds only (no leakage from the validation fold into the transformer).
3. **Baseline modeling** — Logistic Regression, SVM, Random Forest, XGBoost, evaluated on F1 due to class imbalance.
4. **Handling imbalance** — compared SMOTE-based resampling against `class_weight`-based approaches
5. **Hyperparameter tuning** — Bayesian optimisation via `BayesSearchCV` (scikit-optimize), scored on the F1 score.
6. **Feature engineering** — tested combined features and log/square-root transforms on the top candidate
7. **Model selection** — each candidate's threshold set by maximising F1 on its own PR curve; models compared at that threshold, with curve shape checked for stability.
8. **Interpretation** — SHAP values to explain both global feature importance and individual predictions.

## Results

## Results

**Final model:** XGBoost (`class_weight='balanced'`), tuned via `BayesSearchCV`

| Split | F1 Score |
|---|---|
| Train | 0.494 |
| Test  | 0.519 |

Test F1 slightly exceeds train F1, indicating no overfitting.

An F1 of ~0.5 reflects the difficulty of the task: the positive class is ~11% of the data, and `duration` — a feature that would trivially inflate this score — is excluded due to target leakage (it's only known after the call ends).

A feature-engineered variant of this model performed near-identically. The simpler, original-feature version was selected instead for easier SHAP interpretation.

Full comparison across baseline, class-weighted, SMOTE, and tuned models is available in the notebook.
---

## Repository structure

```
Bank_Marketing.ipynb        Main notebook: EDA → modelling → interpretation
best_model/                 Trained model + selected decision threshold, exported for deployment to the backend (bank_marketing_fastapi)
data/                       Raw dataset
demo_input/                 Sample data used by the API and Streamlit app
hyperparameter_results/     BayesSearchCV search logs
requirements/               Pinned dependencies
```

---

## Reproducing

```bash
git clone https://github.com/Itaru2018/bankmarketing_jupyter.git
cd bankmarketing_jupyter
conda env create -f requirements/bank_marketing_jupyter.yml
conda activate bank_marketing_jupyter
jupyter lab Bank_Marketing.ipynb
```

## Stack

Python · pandas · scikit-learn · imbalanced-learn · XGBoost · scikit-optimize · SHAP · matplotlib / seaborn

---

## Author

**Itaru Yasumura** — Basel, Switzerland
[GitHub](https://github.com/Itaru2018) · `https://www.linkedin.com/in/itaru-yasumura-27b05a1b2/`
