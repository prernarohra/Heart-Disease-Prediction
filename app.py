import streamlit as st
import numpy as np
import joblib  

model = joblib.load('trained_model.pkl')

def user_input_features():

    Body_mass_index = st.number_input('BMI', min_value=10, max_value=50)
    Smoking = st.number_input('Smoking (0 = No, 1 = Yes)', min_value=0, max_value=1)
    AlcoholDrinking = st.number_input('Alcohol Drinking (0 = No, 1 = Yes)', min_value=0, max_value=1)
    Stroke = st.number_input('Stroke (0 = No, 1 = Yes)', min_value=0, max_value=1)
    PhysicalHealth = st.number_input('Physical Health (Days)', min_value=0, max_value=30)
    MentalHealth = st.number_input('Mental Health (Days)', min_value=0, max_value=30)
    DiffWalking = st.number_input('Difficulty Walking (0 = No, 1 = Yes)', min_value=0, max_value=1)
    Sex = st.number_input('Sex (0 = Female, 1 = Male)', min_value=0, max_value=1)
    AgeCategory = st.number_input('Age', min_value=10, max_value=90)
    Race = st.number_input('Race (0 = White, 1 = Black, 2 = Asian, 3 = American Indian/Alaskan Native, 5 = Hispanic, 4 = Other)', min_value=0, max_value=5)
    Diabetic = st.number_input('Diabetic (0 = No, 1 = Yes)', min_value=0, max_value=1)
    PhysicalActivity = st.number_input('Physical Activity (0 = No, 1 = Yes)', min_value=0, max_value=1)
    GenHealth = st.number_input('General Health (Good = 0, Poor: 1, Very good: 2, Fair: 3, Excellent: 4)', min_value=0, max_value=4)
    SleepTime = st.number_input('Sleep Time (Hours)', min_value=0, max_value=22)
    Asthma = st.number_input('Asthma (0 = No, 1 = Yes)', min_value=0, max_value=1)
    KidneyDisease = st.number_input('Kidney Disease (0 = No, 1 = Yes)', min_value=0, max_value=1)
    SkinCancer = st.number_input('Skin Cancer (0 = No, 1 = Yes)', min_value=0, max_value=1)


    features = np.array([[Body_mass_index, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, 
                          MentalHealth, DiffWalking, Sex, AgeCategory, Race, Diabetic, 
                          PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease, SkinCancer]])
    return features


st.title("Heart Disease Prediction")

st.write("This app predicts heart disease based on various health factors.")

input_data = user_input_features()

if st.button('Predict'):
    prediction = model.predict(input_data) 
    result = "Positive for Heart Disease" if prediction[0] == 1 else "Negative for Heart Disease"
    st.write(f"Predicted result: {result}")  
