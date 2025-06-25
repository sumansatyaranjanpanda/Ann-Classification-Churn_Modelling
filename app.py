import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import streamlit as st


model=load_model('model.h5')

## load the encoder and scaler
with open('lebel_encoder_gender.pkl','rb') as file:
    lebel_encoder_gender=pickle.load(file)

with open('onehot_encoder_Geograpy.pkl','rb') as file:
    onehot_encoder_Geograpy=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


## streamlit app
st.title('Customer Churn Prediction')

st.subheader("Enter Customer Details:")

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=720)
geography = st.selectbox("Geography", ["Germany", "France", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=20, value=5)
balance = st.number_input("Account Balance", min_value=0.0, value=75000.00)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card?", [1, 0])
is_active_member = st.selectbox("Is Active Member?", [1, 0])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=85000.00)

# Create a dictionary from user inputs
user_input_dict = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary
}

# Convert to DataFrame
input_df = pd.DataFrame([user_input_dict])

if st.button("Predict Churn"):

    input_df['Gender'] = lebel_encoder_gender.transform(input_df['Gender'])

    geo_value = input_df['Geography'].iloc[0]  # Get actual selected value like 'Germany'
    geo_encoded = onehot_encoder_Geograpy.transform([[geo_value]])



    geo_encoded = onehot_encoder_Geograpy.transform([['Geography']])
    geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_Geograpy.get_feature_names_out(['Geography']))

    input_df=pd.concat([input_df.drop('Geography',axis=1),geo_encoded_df],axis=1)

    ### scaler
    scaled_df=scaler.transform(input_df)

    prediction=model.predict(scaled_df)[0][0]
    result = "🚫 Churn" if prediction > 0.5 else "✅ No Churn"

    # Display
    st.markdown(f"### Prediction Probability: `{prediction:.2f}`")
    st.success(f"**Result:** {result}")

