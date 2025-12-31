import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pypdf import PdfReader

# --- 1. CUSTOM FUNCTION (Matches Notebook CELL 4) ---
def to_1d(x):
    if x is None:
        return np.array([''], dtype=object)
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    if isinstance(x, pd.Series):
        x = x.values
    x = np.asarray(x, dtype=object)
    if x.ndim > 1:
        x = x.ravel()
    if x.ndim == 0 or (isinstance(x, np.ndarray) and x.shape == ()):
        x = np.array([str(x.item() if hasattr(x, 'item') else x)], dtype=object)
    x = np.array([str(item) if item is not None else '' for item in x], dtype=object)
    return x

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_models():
    preprocessor = joblib.load("preprocessor.pkl")
    role_model = joblib.load("role_classifier.pkl")
    salary_model = joblib.load("salary_regressor.pkl")
    return preprocessor, role_model, salary_model

try:
    preprocessor, role_model, salary_model = load_models()
except Exception as e:
    st.error(f"Model Load Error: {e}")

# --- 3. UI AND PREDICTION ---
st.title("💼 Job Market Predictor")

uploaded_file = st.file_uploader("Upload CV (PDF)", type="pdf")

if uploaded_file is not None:
    with st.spinner("Analyzing..."):
        # Extract PDF text
        reader = PdfReader(uploaded_file)
        cv_text = ""
        for page in reader.pages:
            cv_text += page.extract_text() or ""

        # --- MATCHING NOTEBOOK CELL 3 & 5 ---
        # We must provide EVERY column used during training
        input_df = pd.DataFrame({
            "Company Size": [500],
            "Avg_Experience": [5.0],
            "Min_Experience": [2.0],        # Missing column fix
            "Max_Experience": [8.0],        # Missing column fix
            "Experience_Range": [6.0],      # Missing column fix (Max - Min)
            "Salary_Range": [20.0],
            "Qualifications": ["Bachelor's Degree"],
            "Work Type": ["Full-time"],
            "Country": ["United States"],   # Missing column fix
            "skills": [cv_text]
        })

        try:
            # Predict
            role = role_model.predict(input_df)[0]
            salary = salary_model.predict(input_df)[0]

            st.success("Results Found!")
            st.metric("Suggested Role", role)
            st.metric("Estimated Salary", f"${salary:.2f}K")
        except Exception as e:
            st.error(f"Prediction Error: {e}")