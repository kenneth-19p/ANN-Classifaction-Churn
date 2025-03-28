import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model
import datetime


### Load the trained model, scaler pickle and onehot
model = load_model('model.h5')

with open('label_gender.pickle', 'rb') as file:
    encoder = pickle.load(file)

with open('scaler.pickle', 'rb') as file:
    scaler = pickle.load(file) 

with open('onehotencoder.pickle', 'rb') as file:
    onehot = pickle.load(file)



## Streamlit app
st.title('Customer Churn Prediction')

#User input

geography = st.selectbox('Geography', onehot.categories_[0])
gender = st.selectbox('Gender', encoder.classes_)
age = st.slider('Age', 18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('CreditScore')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0,10)
num_of_products = st.slider('NumOfProducts', 1,4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Memeber', [0,1])

## Prepare the input data
input_data = pd.DataFrame({
    
    'CreditScore': [credit_score],
    'Gender': [encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]

})

## One-Hot Encode 'Geography'
geo_encoded = onehot.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns =onehot.get_feature_names_out(['Geography']))


## Combine the input data with the one-hot encoded data
input_data = pd.concat([input_data.reset_index(drop = True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

### Prediction churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba: .2f}')

if prediction_proba > 0.5:
   st.write('Customer is likely to churn.')
else:
    st.write('Customer is not likely to churn.')