import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="FinShield | Fraud Detection Engine",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------------------------------------------------------
# Path Configurations
# -------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

MODEL_PATHS = {
    "LightGBM Anomaly Engine": os.path.join(PROCESSED_DIR, "model_lightgbm.pkl"),
    "Random Forest Classifier": os.path.join(PROCESSED_DIR, "model_randomforest.pkl"),
    "Advanced Cyclical Model": os.path.join(PROCESSED_DIR, "model_advancedcyclical.pkl")
}

# Feature Engineering Functions
def add_cyclical_features(df: pd.DataFrame, hour_col: str = "scaled_time") -> pd.DataFrame:
    """Add sine and cosine cyclical features for hour column."""
    df_copy = df.copy()
    if hour_col in df_copy.columns:
        hour_values = df_copy[hour_col].values
        hour_angle = (hour_values % 24) / 24.0 * 2 * np.pi
        df_copy['hour_sin'] = np.sin(hour_angle)
        df_copy['hour_cos'] = np.cos(hour_angle)
    return df_copy


def dynamic_threshold_postprocessing(
    y_score: np.ndarray,
    amounts: np.ndarray,
    historical_avg: float = 1.0,
    base_threshold: float = 0.5,
    alpha: float = 0.15,
) -> np.ndarray:
    """Apply dynamic threshold based on transaction amount using log-scale ratio."""
    historical_avg = max(historical_avg, 1e-6)
    amounts = np.array(amounts)
    amounts = np.where(amounts <= 0, 1e-6, amounts)
    
    log_ratio = np.log(amounts / historical_avg)
    adjusted_threshold = base_threshold * (1 - alpha * np.clip(log_ratio, -5, 5))
    adjusted_threshold = np.clip(adjusted_threshold, 0.1, 0.95)
    
    predictions = (y_score >= adjusted_threshold).astype(int)
    return predictions


# Helper Functions & Model Caching
@st.cache_resource
def load_ml_model(model_path):
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None

# --- Application Header ---
st.title("🛡️ FinShield: High-Stakes Fraud Detection Engine")
st.markdown("""
This production-grade operational dashboard screens credit card transactions for fraudulent anomalies in real time. 
Optimized for **PR-AUC and Recall** to protect fintech revenue pipelines without compromising legitimate customer experiences.
""")
st.write("---")

# --- Sidebar Configuration Panel ---
st.sidebar.header("🛠️ Engine Configurations")
selected_model_name = st.sidebar.selectbox("Select Target ML Architecture", list(MODEL_PATHS.keys()))
model_file_path = MODEL_PATHS[selected_model_name]

model = load_ml_model(model_file_path)
if model is None:
    st.sidebar.error(f"❌ Core model file missing at path:\n`{model_file_path}`")
    st.stop()
else:
    st.sidebar.success(f"⚡ {selected_model_name} loaded successfully.")

risk_threshold = st.sidebar.slider(
    "Custom Risk Threshold Strategy", 
    min_value=0.05, 
    max_value=0.95, 
    value=0.30, 
    step=0.05
)

use_dynamic_threshold = st.sidebar.checkbox(
    "Enable Dynamic Threshold (Transaction Amount-Based)",
    value=selected_model_name == "Advanced Cyclical Model"
)

# Expected and Important Columns
CRITICAL_FEATURES = {
    'merchant_category': 'Categorical',
    'foreign_transaction': 'Binary',
    'location_mismatch': 'Binary',
    'device_trust_score': 'Numeric',
    'scaled_amount': 'Numeric',
    'scaled_time': 'Numeric',
    'velocity_last_24h': 'Numeric',
    'cardholder_age': 'Numeric'
}

OPTIONAL_FEATURES = {
    'hour_sin': 'Cyclical (computed)',
    'hour_cos': 'Cyclical (computed)',
}

# -------------------------------------------------------------------------
# UI Tabs Split
# -------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🔍 Real-Time Transaction Screening", "📂 Automated Batch File Processing"])

with tab1:
    st.subheader("Simulate a Live Transaction Split")
    st.markdown("Adjust the raw transaction features below to simulate inbound payment streaming payloads.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=100000.0, value=85.50, step=1.0)
        transaction_hour = st.slider("Transaction Hour of Day", min_value=0, max_value=23, value=12)
        merchant_category = st.selectbox(
            "Merchant Category", 
            ["shopping", "entertainment", "dining", "groceries", "travel", "utilities", "other"]
        )
        
    with col2:
        foreign_transaction = st.selectbox("Foreign Transaction?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        location_mismatch = st.selectbox("Location Mismatch?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        device_trust_score = st.slider("Device Trust Score", min_value=0, max_value=100, value=85)

    with col3:
        velocity_last_24h = st.number_input("Velocity Score (Trx Last 24h)", min_value=0.0, max_value=100.0, value=1.2, step=0.1)
        cardholder_age = st.number_input("Customer Age", min_value=18, max_value=100, value=35, step=1)

    if st.button("Evaluate Transaction Risk Profile", type="primary"):
        # Explicit Transformer Mapping
        feature_dict = {
            'merchant_category': [merchant_category],
            'foreign_transaction': [foreign_transaction],
            'location_mismatch': [location_mismatch],
            'device_trust_score': [device_trust_score],
            'scaled_amount': [amount],
            'scaled_time': [transaction_hour],
            'velocity_last_24h': [velocity_last_24h],
            'cardholder_age': [cardholder_age]
        }
        
        feature_df = pd.DataFrame(feature_dict)
        
        # Add cyclical features if using Advanced model
        if "Cyclical" in selected_model_name:
            feature_df = add_cyclical_features(feature_df, hour_col="scaled_time")
        
        # Ensure correct column order for model prediction
        expected_cols = [c for c in feature_df.columns if c in list(CRITICAL_FEATURES.keys()) + list(OPTIONAL_FEATURES.keys())]
        feature_df = feature_df[expected_cols]
        
        probabilities = model.predict_proba(feature_df)[0]
        fraud_probability = probabilities[1]
        
        # Apply dynamic threshold if enabled
        if use_dynamic_threshold and 'scaled_amount' in feature_df.columns:
            pred = dynamic_threshold_postprocessing(
                np.array([fraud_probability]),
                feature_df['scaled_amount'].values,
                historical_avg=1.0,
                base_threshold=risk_threshold,
                alpha=0.15
            )[0]
            st.write("---")
            if pred == 1:
                st.error(f"🚨 **HIGH RISK TRANSACTION DETECTED** | Fraud Probability: {fraud_probability * 100:.2f}%")
            else:
                st.success(f"✅ **TRANSACTION AUTHORIZED** | Fraud Probability: {fraud_probability * 100:.2f}%")
        else:
            st.write("---")
            if fraud_probability >= risk_threshold:
                st.error(f"🚨 **HIGH RISK TRANSACTION DETECTED** | Fraud Probability: {fraud_probability * 100:.2f}%")
            else:
                st.success(f"✅ **TRANSACTION AUTHORIZED** | Fraud Probability: {fraud_probability * 100:.2f}%")

with tab2:
    st.subheader("Operational Batch Processing File Sandbox")
    st.markdown("""
    Upload transactional record batch payloads to mass-classify historical records or live trial frames.
    
    **Flexible Schema Support:** While this system works best with all critical features, it can operate with partial data.
    Missing features will be flagged with warnings.
    """)
    
    uploaded_file = st.file_uploader("Drop transaction file formats here (.csv)", type=["csv"], key="batch_uploader")
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            
            st.write("### Raw Data Preview")
            st.dataframe(input_df.head(5), use_container_width=True)
            
            if st.button("Run Bulk Anomaly Pipeline Analysis", type="primary"):
                with st.spinner("Processing transaction matrix..."):
                    
                    eval_df = input_df.copy()
                    eval_df.columns = eval_df.columns.str.strip()
                    
                    # Direct dictionary remap for files that might have variant names
                    rename_map = {
                        'amount': 'scaled_amount',
                        'transaction_hour': 'scaled_time',
                        'hour': 'scaled_time',
                        'velocity_score': 'velocity_last_24h',
                        'customer_age': 'cardholder_age',
                        'age': 'cardholder_age'
                    }
                    eval_df = eval_df.rename(columns=rename_map)
                    
                    # ===================================================================
                    # FEATURE VALIDATION & WARNINGS
                    # ===================================================================
                    st.write("---")
                    st.subheader("📋 Schema Analysis")
                    
                    missing_critical = []
                    available_critical = []
                    
                    for feat, dtype in CRITICAL_FEATURES.items():
                        if feat in eval_df.columns:
                            available_critical.append(feat)
                        else:
                            missing_critical.append(feat)
                    
                    # Display feature availability
                    col_status_a, col_status_b = st.columns(2)
                    with col_status_a:
                        st.success(f"✓ **Available Critical Features** ({len(available_critical)}/{len(CRITICAL_FEATURES)})")
                        for feat in available_critical:
                            st.write(f"   • {feat}")
                    
                    with col_status_b:
                        if missing_critical:
                            st.warning(f"⚠️ **Missing Features** ({len(missing_critical)})")
                            for feat in missing_critical:
                                st.write(f"   • {feat}")
                    
                    # Handle missing features
                    if missing_critical:
                        st.info(f"""
                        **Feature Gap Analysis:**
                        - The model will proceed with available features
                        - Predictions may have reduced accuracy without: {', '.join(missing_critical[:3])}{'...' if len(missing_critical) > 3 else ''}
                        - Consider providing these fields for improved fraud detection
                        """)
                        
                        # Fill missing critical features with reasonable defaults
                        for feat in missing_critical:
                            if feat == 'merchant_category':
                                eval_df[feat] = 'other'
                            elif feat in ['foreign_transaction', 'location_mismatch']:
                                eval_df[feat] = 0
                            elif feat in ['device_trust_score', 'velocity_last_24h', 'cardholder_age']:
                                eval_df[feat] = eval_df[feat].fillna(50) if feat in eval_df.columns else 50
                            elif feat == 'scaled_amount':
                                eval_df[feat] = eval_df[feat].fillna(0) if feat in eval_df.columns else 0
                            elif feat == 'scaled_time':
                                eval_df[feat] = eval_df[feat].fillna(12) if feat in eval_df.columns else 12
                    
                    # Add cyclical features if using Advanced model
                    if "Cyclical" in selected_model_name and 'scaled_time' in eval_df.columns:
                        eval_df = add_cyclical_features(eval_df, hour_col="scaled_time")
                    
                    # Get features for model
                    available_features = [c for c in list(CRITICAL_FEATURES.keys()) + list(OPTIONAL_FEATURES.keys()) 
                                        if c in eval_df.columns]
                    
                    if not available_features:
                        st.error("❌ No recognized features found in the CSV. Cannot proceed with prediction.")
                        st.stop()
                    
                    model_input = eval_df[available_features]
                    
                    # Run predictions
                    batch_probs = model.predict_proba(model_input)[:, 1]
                    
                    # Apply dynamic threshold if enabled
                    if use_dynamic_threshold and 'scaled_amount' in eval_df.columns:
                        batch_predictions = dynamic_threshold_postprocessing(
                            batch_probs,
                            eval_df['scaled_amount'].values,
                            historical_avg=1.0,
                            base_threshold=risk_threshold,
                            alpha=0.15
                        )
                    else:
                        batch_predictions = (batch_probs >= risk_threshold).astype(int)
                    
                    output_df = input_df.copy()
                    output_df['Fraud_Risk_Score'] = batch_probs
                    output_df['System_Action'] = np.where(batch_predictions == 1, "🚫 BLOCK & REVIEW", "✅ APPROVE")
                    
                    flagged_cases = int(np.sum(batch_predictions))
                    total_cases = len(batch_predictions)
                    
                    st.write("---")
                    st.subheader("📊 Batch Settlement Metrics Report")
                    m_col1, m_col2, m_col3 = st.columns(3)
                    m_col1.metric("Total Records Processed", f"{total_cases:,}")
                    m_col2.metric("Flagged Anomaly Exceptions", f"{flagged_cases:,}", 
                                delta=f"{flagged_cases/total_cases*100:.2f}% Flag Rate", delta_color="inverse")
                    
                    estimated_savings = flagged_cases * 145 
                    m_col3.metric("Prevented Fraud Exposure (Est.)", f"${estimated_savings:,}")
                    
                    st.dataframe(output_df, use_container_width=True)
                    
                    csv_data = output_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Flagged Compliance Report",
                        data=csv_data,
                        file_name="finshield_flagged_transactions.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
            import traceback
            st.write(traceback.format_exc())