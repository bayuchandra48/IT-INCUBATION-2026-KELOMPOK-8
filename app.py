
import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------
# Load model & preprocessing hasil Modul 2 & 3
# ------------------------------------------------------------
rf_model = joblib.load("random_forest_model.pkl")
imputer = joblib.load("median_imputer.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Prediksi Final Grade Mahasiswa")
st.write("Aplikasi ini memprediksi `final_grade` (A/B/C/D/F) berdasarkan data akademik dan kebiasaan belajar mahasiswa.")

st.header("Input Data Mahasiswa")

gender = st.selectbox("Gender", ["Male", "Female"])
study_time_hours = st.number_input("Study Time (jam/hari)", min_value=0.0, max_value=24.0, value=3.5, step=0.1)
attendance_percent = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=0.1)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.1)
parental_education = st.selectbox("Parental Education", ["None", "High School", "Bachelors", "Masters", "PhD"])
internet_access = st.selectbox("Internet Access", ["Yes", "No"])
extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
part_time_job = st.selectbox("Part-time Job", ["Yes", "No"])
previous_grade = st.number_input("Previous Grade", min_value=0.0, max_value=100.0, value=75.0, step=0.1)

if st.button("Prediksi"):
    # --------------------------------------------------------
    # Encoding harus identik dengan Modul 2 (Binary Mapping & Ordinal Encoding)
    # --------------------------------------------------------
    input_dict = {
        "gender": {"Male": 0, "Female": 1}[gender],
        "study_time_hours": study_time_hours,
        "attendance_percent": attendance_percent,
        "sleep_hours": sleep_hours,
        "parental_education": {"None": 0, "High School": 1, "Bachelors": 2, "Masters": 3, "PhD": 4}[parental_education],
        "internet_access": {"No": 0, "Yes": 1}[internet_access],
        "extracurricular_activities": {"No": 0, "Yes": 1}[extracurricular_activities],
        "part_time_job": {"No": 0, "Yes": 1}[part_time_job],
        "previous_grade": previous_grade,
    }

    # Create DataFrame directly with feature_columns to ensure correct order and presence
    input_df = pd.DataFrame([input_dict], columns=feature_columns)

    # Imputer yang sama dengan training (tidak di-fit ulang)
    input_imputed = pd.DataFrame(
        imputer.transform(input_df),
        columns=feature_columns
    )

    prediction = rf_model.predict(input_imputed)[0]
    st.success(f"Prediksi Final Grade: **{prediction}**")

    proba = rf_model.predict_proba(input_imputed)[0]
    proba_df = pd.DataFrame({
        "Kelas": rf_model.classes_,
        "Probabilitas": proba
    }).sort_values("Probabilitas", ascending=False)

    st.subheader("Probabilitas Tiap Kelas")
    st.dataframe(proba_df, hide_index=True)
