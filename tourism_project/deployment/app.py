import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="PratapVanka/tourism-model", filename="best_tourism_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Wellness Tourism Package - Potential Buyers Prediction App")
st.write("The Wellness Tourism Package - Potential Buyers Prediction App is an internal tool for Visit with Us company staff. This app predicts potential buyers of the WellnessTourism package supporting the decision making in marketing strategies.")
st.write("Enter the customer details to check whether they are likely to purchase the package.")

# List of numerical features in the dataset
numeric_features = [
    'Age',
    'CityTier',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]

# Collect user input
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=125, value=30)
CityTier = st.selectbox("City Tier (tier of the city where the customer resides)", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (duration of the sales pitch in minutes)", min_value=0.0, value=10.0)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting (number of persons visiting the customer)", min_value=1, value=2, step=1)
NumberOfFollowups = st.number_input("Number of Follow-ups (number of follow-ups made to the customer)", min_value=0, value=1, step=1)
PreferredPropertyStar = st.selectbox("Preferred Property Star (star rating of the property preferred by the customer)", [3, 4, 5])
NumberOfTrips = st.number_input("Number of Trips (number of trips the customer has taken in the past)", min_value=0, value=1)
Passport = st.selectbox("Passport (whether the customer has a passport)", ["Yes", "No"])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score (customer's satisfaction score for the sales pitch)", [1, 2, 3, 4, 5])
OwnCar = st.selectbox("Own Car (whether the customer owns a car)", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (number of children visiting the customer)", min_value=0, value=0, step=1)
MonthlyIncome = st.number_input("Monthly Income (customer's monthly income)", min_value=0.0, value=50000.0)
TypeofContact = st.selectbox("Type of Contact (type of contact made with the customer)", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Occupation (customer's occupation)", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender (customer's gender)", ["Male", "Female"])
ProductPitched = st.selectbox("Product Pitched (product pitched to the customer)", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
MaritalStatus = st.selectbox("Marital Status (customer's marital status)", ["Single", "Divorced", "Married"])
Designation = st.selectbox("Designation (customer's job designation)", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': float(Age),
    'CityTier': CityTier,
    'DurationOfPitch': float(DurationOfPitch),
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': float(NumberOfFollowups),
    'PreferredPropertyStar': float(PreferredPropertyStar),
    'NumberOfTrips': float(NumberOfTrips),
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': float(NumberOfChildrenVisiting),
    'MonthlyIncome': float(MonthlyIncome),
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "A Potential Buyer" if prediction == 1 else "Not a Potential Buyer"
    st.write(f"The customer is **{result}** of the Wellness Tourism Package.")
