cat > ~/ChurnPrediction/src/shap_dashboard.py <<'PY'
import os, joblib, pandas as pd, streamlit as st, shap, matplotlib.pyplot as plt

MODEL_PATH   = os.path.expanduser('~/ChurnPrediction/src/churn_xgb_model.pkl')
FEATURE_PATH = os.path.expanduser('~/ChurnPrediction/src/feature_columns.pkl')
RAW_PATH     = os.path.expanduser('~/airflow/data/new_customers.csv')
OUT_PATH     = os.path.expanduser('~/airflow/data/churn_predictions.csv')

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")
st.title("🔍 Churn Prediction Dashboard")

# ---- load artifacts
model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)  # one-hot columns used in training

# ---- training-like preprocessing
def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # safe fills
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]): df[c] = df[c].fillna(0)
        else: df[c] = df[c].fillna('Unknown')
    # one-hot same as training
    df = pd.get_dummies(df, drop_first=True)
    # align to training columns
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df

# ---- load raw inputs
if not os.path.exists(RAW_PATH):
    st.error(f"Input file not found: {RAW_PATH}")
    st.stop()
raw = pd.read_csv(RAW_PATH)
st.subheader("Raw Inputs (first 20)")
st.dataframe(raw.head(20), use_container_width=True)

# ---- build X exactly like training
try:
    X = preprocess(raw)
except Exception as e:
    st.error(f"Preprocessing failed: {e}")
    st.stop()

# sanity: show any missing/extra columns (should be none)
missing = [c for c in feature_columns if c not in X.columns]
extra   = [c for c in X.columns if c not in feature_columns]
if missing:
    st.error(f"Missing columns after preprocess ({len(missing)}): {missing[:10]} ...")
    st.stop()

# ---- predict
try:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

pred_table = raw.copy()
pred_table['churn_probability'] = y_prob.round(4)
pred_table['churn_prediction']  = y_pred
st.subheader("Predictions (first 20)")
st.dataframe(pred_table.head(20), use_container_width=True)

try: pred_table.to_csv(OUT_PATH, index=False)
except: pass

# ---- SHAP
st.header("🧪 SHAP Explanation for Customer Churn Prediction")
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    st.subheader("Global Feature Impact (Summary)")
    fig1 = plt.figure()
    shap.summary_plot(shap_values.values, X, show=False)
    st.pyplot(fig1, clear_figure=True)

    st.subheader("Explain a Single Customer")
    idx = st.slider("Row index", 0, len(X)-1, 0)
    st.write("Selected row (raw):")
    st.write(raw.iloc[[idx]])
    fig2 = plt.figure()
    shap.plots.bar(shap_values[idx], show=False)
    st.pyplot(fig2, clear_figure=True)

except Exception as e:
    st.error(f"SHAP rendering failed: {e}")
PY


