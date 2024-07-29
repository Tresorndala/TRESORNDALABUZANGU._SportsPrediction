# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pickle
import pandas as pd
import joblib
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load the saved model and scaler
model_path = 'C:\\Users\\user\\Documents\\streamlitdeploy\\GradientBoosting.pkl'
scaler_path = 'path_to_your_scaler/scaler.pkl'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Function to preprocess input data
def preprocess_input(preferred_foot, potential, age, shooting, passing, physic, movement_reactions):
    # One-hot encode the 'preferred_foot' field
    preferred_foot_left = 1 if preferred_foot == 'left' else 0
    preferred_foot_right = 1 if preferred_foot == 'right' else 0
    
    # Create a DataFrame
    input_data = {
        'potential': [potential],
        'age': [age],
        'shooting': [shooting],
        'passing': [passing],
        'physic': [physic],
        'movement_reactions': [movement_reactions],
        'preferred_foot_left': [preferred_foot_left],
        'preferred_foot_right': [preferred_foot_right]
    }
    
    df = pd.DataFrame(input_data)
    
    # Convert to sparse matrix
    sparse_matrix = csr_matrix(df.astype(pd.SparseDtype("float", 0)).sparse.to_coo())
    
    # Standardize features using the same scaler used during training
    X_scaled = scaler.transform(sparse_matrix)
    
    return X_scaled

# Function to handle prediction and display result
def predict_rating(preferred_foot, potential, age, shooting, passing, physic, movement_reactions):
    X_input = preprocess_input(preferred_foot, potential, age, shooting, passing, physic, movement_reactions)
    prediction = model.predict(X_input)[0]
    return prediction

# Streamlit application
def main():
    st.title('Football Player Overall Rating Predictor')
    st.markdown('Enter the details of the football player to predict the overall rating.')

    # Input fields
    preferred_foot = st.selectbox('Preferred Foot', ['left', 'right'])
    potential = st.slider('Potential', min_value=50, max_value=100, value=80)
    age = st.slider('Age', min_value=16, max_value=40, value=25)
    shooting = st.slider('Shooting', min_value=50, max_value=100, value=70)
    passing = st.slider('Passing', min_value=50, max_value=100, value=70)
    physic = st.slider('Physic', min_value=50, max_value=100, value=70)
    movement_reactions = st.slider('Movement Reactions', min_value=50, max_value=100, value=70)

    # Predict button
    if st.button('Predict'):
        prediction = predict_rating(preferred_foot, potential, age, shooting, passing, physic, movement_reactions)
        st.success(f'Predicted Overall Rating: {prediction:.2f}')

if __name__ == '__main__':
    main()

