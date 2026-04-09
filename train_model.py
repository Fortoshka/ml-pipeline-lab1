import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "cars.csv"
CLEAN_DATA_PATH = DATA_DIR / "cars_clean.csv"
MODEL_PATH = DATA_DIR / "cars_model.pkl"
METRICS_PATH = DATA_DIR / "metrics.json"

CATEGORICAL_COLUMNS = ["Make", "Model", "Style", "Fuel_type", "Transmission"]
NUMERIC_COLUMNS = ["Year", "Distance", "Engine_capacity(cm3)", "Price(euro)"]


def ensure_data_dir():
    DATA_DIR.mkdir(exist_ok=True)


def download_data() -> str:
    ensure_data_dir()
    url = (
        "https://raw.githubusercontent.com/dayekb/Basic_ML_Alg/main/cars_moldova_no_dup.csv"
    )
    df = pd.read_csv(url)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"download_data: saved {len(df)} rows to {RAW_DATA_PATH}")
    return str(RAW_DATA_PATH)


def clear_data() -> str:
    ensure_data_dir()
    df = pd.read_csv(RAW_DATA_PATH)
    df = df.dropna(subset=NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)

    df = df[df.Year.between(1971, 2024)]
    df = df[df.Distance.between(0, 1_000_000)]
    df = df[df["Engine_capacity(cm3)"].between(200, 5000)]
    df = df[df["Price(euro)"].between(100, 100_000)]

    ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[CATEGORICAL_COLUMNS] = ordinal.fit_transform(df[CATEGORICAL_COLUMNS])
    df = df.reset_index(drop=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"clear_data: saved {len(df)} rows to {CLEAN_DATA_PATH}")
    return str(CLEAN_DATA_PATH)


def eval_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def train_model() -> str:
    ensure_data_dir()
    df = pd.read_csv(CLEAN_DATA_PATH)
    if df.empty:
        raise ValueError("No training data found. Run clear_data first.")

    X = df.drop(columns=["Price(euro)"])
    y = df["Price(euro)"]

    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scaled = scaler.fit_transform(X.values)
    y_scaled = power_trans.fit_transform(y.values.reshape(-1, 1)).ravel()

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.3, random_state=42
    )

    param_grid = {
        "alpha": [0.0001, 0.001, 0.01],
        "l1_ratio": [0.01, 0.1, 0.2],
        "penalty": ["l1", "l2", "elasticnet"],
        "loss": ["squared_error", "huber"],
        "fit_intercept": [True, False],
    }

    model = SGDRegressor(random_state=42, max_iter=1000, tol=1e-3)
    search = GridSearchCV(model, param_grid, cv=3, n_jobs=4, verbose=0)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred_scaled = best_model.predict(X_val)
    y_pred = power_trans.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_val_orig = power_trans.inverse_transform(y_val.reshape(-1, 1)).ravel()

    metrics = eval_metrics(y_val_orig, y_pred)
    results = {
        "best_params": search.best_params_,
        "metrics": metrics,
        "n_train": int(X_train.shape[0]),
        "n_validation": int(X_val.shape[0]),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(results, metrics_file, indent=2, ensure_ascii=False)

    joblib.dump(best_model, MODEL_PATH)
    print(f"train_model: saved model to {MODEL_PATH}")
    print(f"train_model: metrics {metrics}")
    return str(MODEL_PATH)
