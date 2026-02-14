import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix

# Page Config
st.set_page_config(page_title="ML Classification Dashboard", layout="wide")

st.title("ML Assignment 2: Classification Dashboard")
st.markdown("Implemented by: [Naveen Kumar K / 2025AA05963]")

# Load Models and Scaler
@st.cache_resource
def load_assets():
    models = {}
    model_names = ["Logistic_Regression", "Decision_Tree", "KNN", "Naive_Bayes", "Random_Forest", "XGBoost"]
    for name in model_names:
        models[name.replace("_", " ")] = joblib.load(f"model/{name}.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return models, scaler

try:
    models, scaler = load_assets()
    st.success("Models loaded successfully!")
except FileNotFoundError:
    st.error("Models not found. Please run 'train_models.py' first.")
    st.stop()

# Sidebar - Inputs
st.sidebar.header("User Input Features")

# Dataset upload option
uploaded_file = st.sidebar.file_uploader("Upload your CSV (Test Data)", type=["csv"])

# Model selection dropdown
selected_model_name = st.sidebar.selectbox("Select ML Model", list(models.keys()))

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(df.head())

    # Preprocessing (Assumes last column is target, rest are features)
    try:
        X_test = df.iloc[:, :-1] # All columns except last
        y_test = df.iloc[:, -1]  # Last column is target

        # Scale features
        X_test_scaled = scaler.transform(X_test)

        # Prediction
        model = models[selected_model_name]
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Display evaluation metrics
        st.subheader(f"Performance Metrics: {selected_model_name}")

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        with col1: st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        with col2: st.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.4f}")
        with col3: st.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.4f}")
        with col4: st.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
        with col5: st.metric("AUC Score", f"{roc_auc_score(y_test, y_pred_proba):.4f}" if len(np.unique(y_test)) == 2 else "N/A (Multi-class)")
        with col6: st.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")

        # Confusion Matrix 
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing data: {e}. Ensure your CSV matches the training data format.")

else:
    st.info("Please upload a CSV file to generate predictions.")
