# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:35:46 2024

@author: user
"""
import joblib
import streamlit as st
import numpy as np

# Define the path to your .pkl file
model_path = r"C:\Users\user\Documents\streamlitdeploy\GradientBoosting (3).pkl"

# Load the model
try:
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    st.write("Model loaded successfully!")
    # Optionally, you can print or inspect the loaded model
    st.write("Loaded model:", model)
    
    # Function to preprocess input data
    def preprocess_input(preferred_foot, potential, age, shooting, passing, physic, movement_reactions):
        # Create a list with the input features in the required order
        preferred_foot_binary = 1 if preferred_foot == 'left' else 0
        
        input_data = [
            potential, age, shooting, passing, physic, movement_reactions,
            preferred_foot_binary
        ]
        
        return np.array(input_data).reshape(1, -1)  # Reshape to a 2D array

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

except FileNotFoundError:
    st.error(f"Error: The file '{model_path}' was not found.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
