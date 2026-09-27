import streamlit as st
import pandas as pd
import joblib
# Load model dan preprocessing
rf_model = joblib.load("random_forest_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
st.title("Student Performance Prediction")
st.write(
 "Aplikasi demo Machine Learning menggunakan "
 "Random Forest Classifier."
)
# Input pengguna
gender = st.selectbox("Gender", ["Male", "Female"])
previous_grade = st.number_input(
 "Previous Grade", min_value=60.0, max_value=90.0, value=78.0
)
extracurricular = st.number_input(
 "Extracurricular Activities", min_value=0, max_value=3, value=1
)
parental_support = st.selectbox(
 "Parental Support", ["Low", "Medium", "High"]
)
study_hours = st.number_input(
 "Study Hours", min_value=0.0, max_value=5.0, value=2.5
)
attendance = st.number_input(
 "Attendance (%)", min_value=50.0, max_value=100.0, value=76.0
)
online_classes = st.selectbox(
 "Online Classes Taken", ["No", "Yes"]
)
if st.button("Predict"):
 gender_mapping = {"Male": 0, "Female": 1}
 parental_mapping = {"Low": 0, "Medium": 1, "High": 2}
 online_mapping = {"No": 0, "Yes": 1}
 input_data = pd.DataFrame([{
 "Gender": gender_mapping[gender],
 "PreviousGrade": previous_grade,
 "ExtracurricularActivities": extracurricular,
 "ParentalSupport": parental_mapping[parental_support],
 "Study Hours": study_hours,
 "Attendance (%)": attendance,
 "Online Classes Taken": online_mapping[online_classes]
 }])
 # Pastikan urutan feature sama seperti training
 input_data = input_data[feature_columns]
 # Gunakan imputer yang sudah di-fit pada training data
 input_imputed = pd.DataFrame(
 imputer.transform(input_data),
 columns=feature_columns
 )
 prediction = rf_model.predict(input_imputed)
 st.subheader("Hasil Prediksi")
 st.write(f"Predicted Status: {prediction[0]}")