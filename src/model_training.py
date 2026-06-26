import os
from typing import Dict, Tuple, Callable

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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


def load_processed_data(processed_dir: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).iloc[:, 0]
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).iloc[:, 0]
    return X_train, y_train, X_test, y_test


def add_cyclical_features(X: pd.DataFrame, hour_col: str = "scaled_time") -> pd.DataFrame:
    """
    Transform the hour column into cyclical features using Sine and Cosine transformations.
    
    This captures the cyclic nature of time on a 24-hour clock. The hours are mapped
    to angles in the unit circle, preserving proximity relationships (e.g., hour 23 and
    hour 1 are close neighbors, not far apart).
    
    Args:
        X: DataFrame containing the hour column
        hour_col: Name of the column containing hour values (0-23)
        
    Returns:
        DataFrame with original data plus new cyclical features:
        - hour_sin: Sine component of the hour in 24-hour cycle
        - hour_cos: Cosine component of the hour in 24-hour cycle
    """
    X_with_cycles = X.copy()
    
    # Convert hour to angle in radians (24-hour cycle)
    # hour_angle = (hour / 24) * 2π
    hour_values = X_with_cycles[hour_col].values
    hour_angle = (hour_values % 24) / 24.0 * 2 * np.pi
    
    # Create cyclical features
    X_with_cycles['hour_sin'] = np.sin(hour_angle)
    X_with_cycles['hour_cos'] = np.cos(hour_angle)
    
    return X_with_cycles


def compute_historical_average(X_train: pd.DataFrame, amount_col: str = "scaled_amount") -> float:
    """
    Compute historical average transaction amount from training data.
    
    Args:
        X_train: Training data
        amount_col: Column name for transaction amounts
        
    Returns:
        Mean transaction amount
    """
    return X_train[amount_col].mean()


def dynamic_threshold_postprocessing(
    y_score: np.ndarray,
    amounts: np.ndarray,
    historical_avg: float,
    base_threshold: float = 0.5,
    alpha: float = 0.15,
) -> np.ndarray:
    """
    Apply dynamic threshold adjustment based on transaction amount using log-scale ratio.
    
    The threshold is dynamically lowered for transactions with higher amounts relative to
    historical average. This reflects real fraud patterns: unusual amounts are more likely
    to be fraudulent.
    
    Formula: adjusted_threshold = base_threshold * (1 - alpha * log(amount / historical_avg))
    
    Where:
    - base_threshold: Initial classification threshold (e.g., 0.5)
    - alpha: Sensitivity parameter controlling how much the threshold adjusts (0-1)
    - log(amount / historical_avg): Log-scale ratio with historical average
    
    Args:
        y_score: Fraud probability scores from model (0-1)
        amounts: Transaction amounts (must be non-zero)
        historical_avg: Historical average transaction amount
        base_threshold: Base threshold for classification (default 0.5)
        alpha: Sensitivity factor for threshold adjustment (default 0.15)
        
    Returns:
        Binary predictions (0 or 1) after dynamic threshold application
    """
    # Avoid division by zero
    historical_avg = max(historical_avg, 1e-6)
    amounts = np.array(amounts)
    amounts = np.where(amounts <= 0, 1e-6, amounts)  # Handle zero/negative amounts
    
    # Calculate log-scale ratio: log(amount / historical_avg)
    log_ratio = np.log(amounts / historical_avg)
    
    # Compute adjusted threshold for each transaction
    # Higher amounts → lower threshold → more likely to flag as fraud
    adjusted_threshold = base_threshold * (1 - alpha * np.clip(log_ratio, -5, 5))
    adjusted_threshold = np.clip(adjusted_threshold, 0.1, 0.95)  # Keep threshold reasonable
    
    # Apply dynamic threshold to predictions
    predictions = (y_score >= adjusted_threshold).astype(int)
    
    return predictions


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
        "AdvancedCyclical": Pipeline(
            [
                ("preprocessor", None),
                (
                    "classifier",
                    LGBMClassifier(
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                        verbosity=-1,
                        n_estimators=300,
                        learning_rate=0.05,
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
    use_dynamic_threshold: bool = False,
    amounts: np.ndarray = None,
    historical_avg: float = None,
) -> Dict[str, float]:
    """
    Evaluate model with optional dynamic threshold post-processing.
    
    Args:
        name: Model name
        pipeline: Trained pipeline
        X_test: Test features
        y_test: Test labels
        use_dynamic_threshold: Whether to apply dynamic threshold
        amounts: Transaction amounts (required if use_dynamic_threshold=True)
        historical_avg: Historical average amount (required if use_dynamic_threshold=True)
        
    Returns:
        Dictionary of metrics
    """
    y_score = pipeline.predict_proba(X_test)[:, 1]
    
    if use_dynamic_threshold and amounts is not None and historical_avg is not None:
        y_pred = dynamic_threshold_postprocessing(
            y_score,
            amounts,
            historical_avg,
            base_threshold=0.5,
            alpha=0.15,
        )
    else:
        y_pred = pipeline.predict(X_test)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, y_score)

    print(f"\n=== {name} Evaluation ===")
    if use_dynamic_threshold:
        print(f"(Using Dynamic Threshold Post-Processing)")
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

    # =========================================================================
    # FEATURE ENGINEERING: Add Cyclical Features for Advanced Model
    # =========================================================================
    print("\n🔄 Engineering cyclical features for Hour column...")
    X_train_with_cycles = add_cyclical_features(X_train, hour_col="scaled_time")
    X_test_with_cycles = add_cyclical_features(X_test, hour_col="scaled_time")
    
    # Compute historical average for dynamic threshold post-processing
    historical_avg_amount = compute_historical_average(X_train_with_cycles, amount_col="scaled_amount")
    print(f"   ✓ Historical average transaction amount: {historical_avg_amount:.4f}")
    print(f"   ✓ Added cyclical features: hour_sin, hour_cos")
    
    # =========================================================================
    # MODEL BUILDING AND TRAINING
    # =========================================================================
    preprocessor = build_preprocessor(X_train)
    preprocessor_cyclical = build_preprocessor(X_train_with_cycles)
    
    models = build_models()

    # Apply standard preprocessor to first two models
    models["RandomForest"].steps[0] = ("preprocessor", preprocessor)
    models["LightGBM"].steps[0] = ("preprocessor", preprocessor)
    
    # Apply cyclical-aware preprocessor to the advanced model
    models["AdvancedCyclical"].steps[0] = ("preprocessor", preprocessor_cyclical)

    print("\n🧠 Training ensemble models with imbalanced-aware metrics...")
    for name, pipeline in models.items():
        if name == "AdvancedCyclical":
            # Train on cyclical features
            pipeline.fit(X_train_with_cycles, y_train)
            print(f"\n📊 {name} (trained on cyclical features)")
            cv_scores = cross_validate_model(pipeline, X_train_with_cycles, y_train)
        else:
            # Train on standard features
            pipeline.fit(X_train, y_train)
            print(f"\n📊 {name}")
            cv_scores = cross_validate_model(pipeline, X_train, y_train)
        
        print(f"   Cross-validated PR-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Persist trained pipeline for use by the Streamlit app
        model_path = os.path.join(processed_dir, f"model_{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(pipeline, model_path)
        print(f"   Saved to {model_path}")

    # =========================================================================
    # EVALUATION WITH METRICS
    # =========================================================================
    print("\n📈 Evaluating models on test set...")
    metrics = {}
    
    # Evaluate RandomForest and LightGBM with standard approach
    for name in ["RandomForest", "LightGBM"]:
        metrics[name] = evaluate_model(
            name,
            models[name],
            X_test,
            y_test,
            use_dynamic_threshold=False,
        )
    
    # Evaluate AdvancedCyclical with dynamic threshold post-processing
    metrics["AdvancedCyclical"] = evaluate_model(
        "AdvancedCyclical (with Dynamic Threshold)",
        models["AdvancedCyclical"],
        X_test_with_cycles,
        y_test,
        use_dynamic_threshold=True,
        amounts=X_test_with_cycles["scaled_amount"].values,
        historical_avg=historical_avg_amount,
    )

    # =========================================================================
    # LEADERBOARD & VISUALIZATION
    # =========================================================================
    leaderboard_path = os.path.join(processed_dir, "phase3_leaderboard.csv")
    lb_rows = []
    for name, m in metrics.items():
        lb_rows.append({
            "model": name,
            "precision": m["precision"],
            "recall": m["recall"],
            "pr_auc": m["pr_auc"]
        })
    pd.DataFrame(lb_rows).to_csv(leaderboard_path, index=False)
    print(f"\n✓ Saved leaderboard to {leaderboard_path}")

    # Plot PR curves for standard models
    plot_path = os.path.join(processed_dir, "phase3_pr_curve.png")
    plot_pr_curves(
        {k: v for k, v in models.items() if k != "AdvancedCyclical"},
        X_test,
        y_test,
        plot_path
    )

    print("\n✅ Phase 3 completed successfully!")
    print("   🏆 Models trained and ready for deployment")
    print(f"   📊 Leaderboard saved to {leaderboard_path}")
    print(f"   📈 PR curves saved to {plot_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_dir = os.path.normpath(os.path.join(base_dir, "..", "data", "processed"))
    run_phase_3(processed_data_dir)
