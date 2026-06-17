import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

def test_imbalance_strategies(processed_dir: str):
    """
    Loads training data, sets up a cross-validation pipeline,
    and isolates SMOTE to prevent data leakage.
    """
    print("🔄 Loading data splits from Phase 1...")
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).values.ravel()

    print(f"   | Initial training shape: {X_train.shape[0]:,} samples")
    print(f"   | Initial fraud instances: {np.sum(y_train)} ({np.sum(y_train)/len(y_train)*100:.4f}%)")

    # -------------------------------------------------------------------------
    # Skeptical Design: Cross-Validation Setup
    # Using 5-Fold Stratified CV to maintain class distribution across splits
    # -------------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Strategy A: Hybrid Approach (SMOTE up to 10% minority, then Under-sample majority)
    # This keeps the synthetic data realistic while reducing memory/compute strain.
    hybrid_pipeline = ImbPipeline([
        ('smote', SMOTE(sampling_strategy=0.1, random_state=42)),
        ('under', RandomUnderSampler(sampling_strategy=0.5, random_state=42))
    ])

    print("\n🧪 Simulating isolated resampling across Cross-Validation folds...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train[train_idx]
        X_fold_val, y_fold_val = X_train.iloc[val_idx], y_train[val_idx]
        
        # ---------------------------------------------------------------------
        # Preprocess categorical columns and drop identifiers before resampling.
        # Categorical columns must be encoded when using imbalanced-learn resamplers.
        # ---------------------------------------------------------------------
        feature_cols = X_fold_train.columns.tolist()
        if 'transaction_id' in feature_cols:
            feature_cols.remove('transaction_id')

        X_fold_train_enc = pd.get_dummies(X_fold_train[feature_cols], drop_first=True)
        X_fold_val_enc = pd.get_dummies(X_fold_val[feature_cols], drop_first=True)

        # Ensure validation set has same encoded columns as training fold
        X_fold_val_enc = X_fold_val_enc.reindex(columns=X_fold_train_enc.columns, fill_value=0)

        # Apply the resampling pipeline strictly to the training fold
        X_resampled, y_resampled = hybrid_pipeline.fit_resample(X_fold_train_enc, y_fold_train)
        
        print(f"\n   📍 Fold {fold}:")
        print(f"      ├── Pre-Resample  -> Samples: {len(X_fold_train):,}, Fraud: {np.sum(y_fold_train)}")
        print(f"      ├── Post-Resample -> Samples: {len(X_resampled):,}, Fraud: {np.sum(y_resampled)} ({np.sum(y_resampled)/len(y_resampled)*100:.1f}%)")
        print(f"      └── Validation   -> Samples: {len(X_fold_val):,}, Fraud: {np.sum(y_fold_val)} (Untouched/Clean)")

    print("\n✅ Verification complete: Synthetic data generation is completely contained.")
    print("🚀 Ready for Phase 3: Model Training & Metric Optimization.")
    
    return hybrid_pipeline

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DATA_DIR = os.path.normpath(os.path.join(base_dir, "..", "data", "processed"))
    test_imbalance_strategies(PROCESSED_DATA_DIR)