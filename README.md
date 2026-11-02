# RetailMind AI

RetailMind AI predicts customer churn so teams can identify at-risk customers early. The project compares multiple classifiers on the same prepared features, prioritizes recall for the churn class, and persists the selected model as a reusable prediction artifact.

## Project Structure

```
RetailMind AI/
├── data/
│   └── Telco-Customer-Churn.csv          # Training dataset (7,043 rows, 21 columns)
├── models/
│   ├── churn_pipeline.joblib             # Logistic Regression baseline
│   └── champion_churn_pipeline.joblib    # Selected champion (Linear SVC)
├── notebooks/
│   ├── train.ipynb                       # Data exploration, baseline training, evaluation, model export
│   ├── predict.ipynb                     # Loads the baseline model and runs predictions
│   └── predict_cmp.ipynb                 # Model comparison and champion selection
├── src/                                  # Placeholder for production code
├── requirements.txt                      # Python dependencies
└── README.md
```

## Dataset

The project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Each row represents a customer with demographic, account, and service information, plus a `Churn` label (`Yes`/`No`).

Key fields:

- `tenure`, `MonthlyCharges`, `TotalCharges` — account and billing details
- `Contract`, `PaymentMethod`, `InternetService`, etc. — service details
- `Churn` — target variable

## Preprocessing

All models use the same feature preparation:

- `TotalCharges` is coerced to numeric; its 11 blank values become missing and are median-imputed.
- `customerID` is dropped.
- `Churn` is encoded as `Yes → 1`, `No → 0`.
- Numeric features: median imputation + `StandardScaler`.
- Categorical features: `OneHotEncoder(handle_unknown='ignore')`.

Keeping this preprocessing identical ensures every model is compared on the same features.

## Baseline Model

The baseline pipeline uses:

- Numeric features: median imputation + `StandardScaler`
- Categorical features: `OneHotEncoder(handle_unknown='ignore')`
- Classifier: `LogisticRegression(max_iter=4000, random_state=33)`

### Baseline Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 0.7984 |
| Precision | 0.6366 |
| Recall | 0.5589 |
| ROC-AUC | 0.8389 |

Metrics were measured on a 25% stratified test split (`random_state=67`).

## Model Comparison

Three additional classifiers were trained on the same preprocessed features and evaluated on a single shared hold-out set:

- Split: 80/20 stratified hold-out (`random_state=42`)
- Classifiers: Decision Tree, Random Forest, Linear SVM
- Each classifier uses `random_state=42` for reproducibility.

| Model | Accuracy | Precision | Recall (Churn) | ROC-AUC |
|-------|----------|-----------|----------------|---------|
| Decision Tree | 0.7913 | 0.6342 | 0.5053 | 0.8351 |
| **Linear SVC (champion)** | **0.7878** | **0.6176** | **0.5267** | **0.8252** |
| Random Forest | 0.7779 | 0.6034 | 0.4759 | 0.8162 |

Models were ranked by churn recall first, then ROC-AUC as a tie-breaker. Catching at-risk customers is the priority, so recall on the churn class (`Churn = 1`) matters more than raw accuracy.

**Why Linear SVC won:** it achieved the highest churn recall among the three challengers (`0.5267`), ahead of Decision Tree (`0.5053`) and Random Forest (`0.4759`). It gives up a small amount of accuracy and ROC-AUC compared with Decision Tree, which is an acceptable trade-off for identifying more customers likely to churn.

The champion pipeline is persisted at `models/champion_churn_pipeline.joblib` alongside the baseline model.

> Note: The baseline Logistic Regression recorded a higher recall (`0.5589`), but it was evaluated on a different 75/25 split. The comparison above is strictly between the three challengers on the shared 80/20 hold-out.

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter and open the notebooks:

   ```bash
   jupyter notebook
   ```

## Usage

### Train and save the baseline

Open `notebooks/train.ipynb` and run all cells. The trained pipeline is saved to `models/churn_pipeline.joblib`.

### Compare models and save the champion

Open `notebooks/predict_cmp.ipynb` and run all cells. The notebook evaluates Decision Tree, Linear SVC, and Random Forest on the same hold-out set and saves the highest-recall model to `models/champion_churn_pipeline.joblib`.

### Predict

Load either persisted pipeline and generate predictions:

```python
import joblib
import pandas as pd

pipeline = joblib.load("models/champion_churn_pipeline.joblib")
data = pd.read_csv("data/Telco-Customer-Churn.csv").drop(columns=["customerID", "Churn"])

predictions = pipeline.predict(data)
probabilities = pipeline.predict_proba(data)[:, 1]
```

To use the baseline instead, load `models/churn_pipeline.joblib`.

## Notes

- `src/` currently has no production code; training, comparison, and inference logic live in the notebooks.
- Preprocessing is duplicated between the notebooks. Refactoring into a shared `src/` module is a natural next step.
