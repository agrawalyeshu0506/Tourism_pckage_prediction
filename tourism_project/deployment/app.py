import os
import streamlit as st
import pandas as pd
import joblib


# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Wellness Tourism Package Prediction")
st.write("Predict whether a customer will purchase the Wellness Tourism Package.")

# Collect inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10)
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
num_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=2)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe"])
preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_trips = st.number_input("Number of Trips", min_value=0, max_value=20, value=1)
passport = st.selectbox("Passport", [0, 1])
pitch_score = st.number_input("Pitch Satisfaction Score", min_value=0, max_value=5, value=3)
own_car = st.selectbox("Own Car", [0, 1])
num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager"])
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=20000)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    # Build input DataFrame
    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": typeof_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_person_visiting,
        "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": num_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Customer is likely to purchase (probability: {prob:.2f})")
    else:
        st.error(f"Customer is unlikely to purchase (probability: {prob:.2f})")


