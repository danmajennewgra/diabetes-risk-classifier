import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="Early Health Risk Classifier", page_icon="🩺")

st.title("🩺 Smart Assistant: Early Health Risk Classifier")
st.write("Adjust patient metrics using the sliders to run instant AI health evaluations.")

# Load saved model artifacts
@st.cache_resource
def load_artifacts():
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('knn_diabetes_model.pkl')
    return scaler, model

scaler, model = load_artifacts()

# Input Sliders
pregnancies = st.slider("Pregnancies", 0, 17, 1)
glucose = st.slider("Glucose Level (mg/dL)", 40, 200, 120)
bp = st.slider("Blood Pressure (mm Hg)", 40, 122, 70)
skin = st.slider("Skin Thickness (mm)", 7, 99, 20)
insulin = st.slider("Insulin (mu U/ml)", 14, 846, 79)
bmi = st.slider("BMI", 18.0, 67.0, 25.0)
pedigree = st.slider("Diabetes Pedigree Function", 0.07, 2.42, 0.37)
age = st.slider("Age", 21, 81, 33)

if st.button("Predict Risk", type="primary"):
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_data = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]], columns=features)
    
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    probabilities = model.predict_proba(scaled_data)[0]
    
    st.divider()
    if prediction == 1:
        st.error(f"⚠️ **HIGH RISK of Diabetes** (Confidence: {probabilities[1]*100:.1f}%)")
    else:
        st.success(f"✅ **LOW RISK of Diabetes** (Confidence: {probabilities[0]*100:.1f}%)")
