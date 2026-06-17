import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib


def load_processed_data(processed_dir: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).iloc[:, 0]
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).iloc[:, 0]
    return X_train, y_train, X_test, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    columns = X.columns.tolist()
    if "transaction_id" in columns:
        columns.remove("transaction_id")

    # Include explicit 'string' dtype to be compatible with pandas 3/4 string migration
    categorical = X[columns].select_dtypes(include=["object", "category", "string"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="passthrough",
    )


def build_models() -> Dict[str, Pipeline]:
    return {
        "RandomForest": Pipeline(
            [
                ("preprocessor", None),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=250,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "LightGBM": Pipeline(
            [
                ("preprocessor", None),
                (
                    "classifier",
                    LGBMClassifier(
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
    }


def cross_validate_model(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv: int = 5
) -> np.ndarray:
    return cross_val_score(
        pipeline,
        X,
        y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring="average_precision",
        n_jobs=-1,
    )


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, y_score)

    print(f"\n=== {name} Evaluation ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    return {
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
    }


def plot_pr_curves(
    trained_models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str,
) -> None:
    plt.figure(figsize=(10, 7))

    for name, pipeline in trained_models.items():
        PrecisionRecallDisplay.from_estimator(
            pipeline,
            X_test,
            y_test,
            name=name,
            ax=plt.gca(),
        )

    plt.title("Phase 3: Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved PR curve plot to {output_path}")


def run_phase_3(processed_dir: str) -> None:
    print("🔄 Phase 3: Loading processed training and test splits...")
    X_train, y_train, X_test, y_test = load_processed_data(processed_dir)

    if "transaction_id" in X_train.columns:
        X_train = X_train.drop(columns=["transaction_id"])
    if "transaction_id" in X_test.columns:
        X_test = X_test.drop(columns=["transaction_id"])

    preprocessor = build_preprocessor(X_train)
    models = build_models()

    for name, pipeline in models.items():
        pipeline.steps[0] = ("preprocessor", preprocessor)

    print("\n🧠 Training ensemble models with imbalanced-aware metrics...")
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        cv_scores = cross_validate_model(pipeline, X_train, y_train)
        print(f"\n{name} cross-validated PR-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        # Persist trained pipeline for use by the Streamlit app
        model_path = os.path.join(processed_dir, f"model_{name.lower()}.pkl")
        joblib.dump(pipeline, model_path)
        print(f"Saved trained model to {model_path}")

    metrics = {}
    for name, pipeline in models.items():
        metrics[name] = evaluate_model(name, pipeline, X_test, y_test)

    # Save a simple leaderboard
    leaderboard_path = os.path.join(processed_dir, "phase3_leaderboard.csv")
    lb_rows = []
    for name, m in metrics.items():
        lb_rows.append({"model": name, "precision": m["precision"], "recall": m["recall"], "pr_auc": m["pr_auc"]})
    pd.DataFrame(lb_rows).to_csv(leaderboard_path, index=False)
    print(f"Saved leaderboard to {leaderboard_path}")

    plot_path = os.path.join(processed_dir, "phase3_pr_curve.png")
    plot_pr_curves(models, X_test, y_test, plot_path)

    print("\n✅ Phase 3 completed. Focused metrics saved and models are ready for selection.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_dir = os.path.normpath(os.path.join(base_dir, "..", "data", "processed"))
    run_phase_3(processed_data_dir)
