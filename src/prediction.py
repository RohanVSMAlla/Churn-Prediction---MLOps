import pandas as pd
import joblib
import os

def predict_churn():
    # Paths
    model_path = os.path.expanduser('~/ChurnPrediction/src/churn_xgb_model.pkl')
    features_path = os.path.expanduser('~/ChurnPrediction/src/feature_columns.pkl')
    input_path = os.path.expanduser('~/airflow/data/new_customers.csv')
    output_path = os.path.expanduser('~/airflow/data/churn_predictions.csv')

    # Load model and features
    model = joblib.load(model_path)
    feature_columns = joblib.load(features_path)

    # Load new data
    df = pd.read_csv(input_path)

    # One-hot encode new data
    df_encoded = pd.get_dummies(df)

    # Align encoded columns with training features
    missing_cols = set(feature_columns) - set(df_encoded.columns)
    for col in missing_cols:
        df_encoded[col] = 0

    # Reorder to match training feature order
    df_encoded = df_encoded[feature_columns]

    # Predict
    X_new = df_encoded
    preds = model.predict(X_new)

    # Save predictions
    df['Churn_Prediction'] = preds
    df.to_csv(output_path, index=False)
    print("✅ Predictions saved to churn_predictions.csv")


