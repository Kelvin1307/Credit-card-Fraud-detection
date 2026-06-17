import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def run_data_pipeline(data_path: str, output_dir: str):
    """
    Loads the credit card fraud dataset, analyzes class imbalance,
    scales non-PCA features using RobustScaler, and splits data safely.
    """
    print("🔄 Step 1: Loading raw dataset...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Please download it from Kaggle.")
        
    df = pd.read_csv(data_path)

    # -------------------------------------------------------------------------
    # Schema Adaptation: handle alternative column names from different sources
    # Map common variants to the names expected later in the pipeline.
    # -------------------------------------------------------------------------
    def _find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        # case-insensitive fallback
        lower_map = {col.lower(): col for col in df.columns}
        for c in candidates:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        return None

    # Map target columns to expected names
    class_col = _find_col(["Class", "is_fraud", "isFraud", "fraud", "isfraud"])
    amount_col = _find_col(["Amount", "amount", "amt"])
    time_col = _find_col(["Time", "time", "transaction_hour", "hour"])

    if class_col and class_col != "Class":
        df.rename(columns={class_col: "Class"}, inplace=True)
    if amount_col and amount_col != "Amount":
        df.rename(columns={amount_col: "Amount"}, inplace=True)
    if time_col and time_col != "Time":
        df.rename(columns={time_col: "Time"}, inplace=True)

    # Verify required columns exist now
    required = {"Class", "Amount", "Time"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Required columns missing after schema mapping: {sorted(missing)}. Available columns: {list(df.columns)[:20]}")
    
    # -------------------------------------------------------------------------
    # Skeptical Check: Verify Class Imbalance & Missing Values
    # -------------------------------------------------------------------------
    print("\n🔍 Step 2: Running sanity checks and baseline profiling...")
    total_records = len(df)
    fraud_count = df['Class'].sum()
    legit_count = total_records - fraud_count
    fraud_percentage = (fraud_count / total_records) * 100
    
    print(f"   | Total Transactions: {total_records:,}")
    print(f"   | Legitimate (Class 0): {legit_count:,}")
    print(f"   | Fraudulent (Class 1): {fraud_count:,} ({fraud_percentage:.4f}%)")
    
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"   | ⚠️ Warning: Found {missing_values} missing values. Filling with 0.")
        df.fillna(0, inplace=True)
    else:
        print("   | No missing values detected. Dataset is clean.")

    # -------------------------------------------------------------------------
    # Feature Engineering: Scaling Time and Amount
    # V1-V28 are already PCA-transformed. Time and Amount are raw and skewed.
    # We use RobustScaler because fraud amounts contain extreme outliers.
    # -------------------------------------------------------------------------
    print("\n⚖️ Step 3: Scaling 'Time' and 'Amount' features using RobustScaler...")
    scaler = RobustScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    # Drop original unscaled columns
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    # -------------------------------------------------------------------------
    # Data Splitting
    # CRITICAL: We do NOT apply SMOTE/resampling here. 
    # Resampling must strictly happen inside cross-validation or on training data only.
    # -------------------------------------------------------------------------
    print("\n✂️ Step 4: Stratifying and splitting data into Train and Test sets...")
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Stratify=y ensures both splits maintain the exact 0.17% fraud ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   | Training set size: {X_train.shape[0]:,} samples")
    print(f"   | Testing set size: {X_test.shape[0]:,} samples")
    print(f"   | Test Set Fraud Ratio: {y_test.sum() / len(y_test) * 100:.4f}% (Perfect Stratification)")

    # -------------------------------------------------------------------------
    # Exporting Splits
    # -------------------------------------------------------------------------
    print(f"\n💾 Step 5: Exporting processed splits to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False, header=True)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False, header=True)
    
    print("✅ Phase 1 Pipeline completed successfully! Ready for Imbalance Handling.")

if __name__ == "__main__":
    # Adjust paths if your file names differ
    # Prefer common expected filenames, fall back to any CSV inside `data/`
    candidate_paths = [
        "data/creditcard.csv",
        "data/credit_card_fraud_10k.csv",
        "data/creditcard_10k.csv",
    ]

    RAW_DATA_PATH = None
    for p in candidate_paths:
        if os.path.exists(p):
            RAW_DATA_PATH = p
            break

    if RAW_DATA_PATH is None:
        csvs = glob.glob(os.path.join("data", "*.csv"))
        if csvs:
            RAW_DATA_PATH = csvs[0]
            print(f"ℹ️ Detected data file: {RAW_DATA_PATH}")
        else:
            # Let the pipeline raise a clear FileNotFoundError as before
            RAW_DATA_PATH = "data/creditcard.csv"

    OUTPUT_DIRECTORY = "data/processed"

    run_data_pipeline(RAW_DATA_PATH, OUTPUT_DIRECTORY)