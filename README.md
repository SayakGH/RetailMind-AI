# RetailMind AI

A simple customer churn prediction project built on the IBM Telco Customer Churn dataset. It trains a scikit-learn pipeline and uses the saved model to predict whether a customer is likely to churn.

## Project Structure

```
RetailMind AI/
├── data/
│   └── Telco-Customer-Churn.csv   # Training dataset (7,043 rows, 21 columns)
├── models/
│   └── churn_pipeline.joblib      # Trained and persisted model pipeline
├── notebooks/
│   ├── train.ipynb                # Data exploration, training, evaluation, model export
│   └── predict.ipynb              # Loads the model and runs predictions
├── src/                           # Placeholder for production code
├── requirements.txt               # Python dependencies
└── README.md
```

## Dataset

The project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Each row represents a customer with demographic, account, and service information, plus a `Churn` label (`Yes`/`No`).

Key fields:

- `tenure`, `MonthlyCharges`, `TotalCharges` — account and billing details
- `Contract`, `PaymentMethod`, `InternetService`, etc. — service details
- `Churn` — target variable

## Model Pipeline

The training notebook builds a scikit-learn `Pipeline` with:

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

Metrics were measured on a 25% stratified test split.

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

### Train and save the model

Open `notebooks/train.ipynb` and run all cells. The trained pipeline is saved to `models/churn_pipeline.joblib`.

### Predict

Open `notebooks/predict.ipynb` to load the saved pipeline and generate predictions:

```python
import joblib
import pandas as pd

pipeline = joblib.load("models/churn_pipeline.joblib")
data = pd.read_csv("data/Telco-Customer-Churn.csv").drop(columns=["customerID", "Churn"])

predictions = pipeline.predict(data)
probabilities = pipeline.predict_proba(data)[:, 1]
```

## Notes

- `src/` currently has no production code; training and inference logic live in the notebooks.
- Preprocessing is duplicated between the notebooks. Refactoring into a shared `src/` module is a natural next step.
