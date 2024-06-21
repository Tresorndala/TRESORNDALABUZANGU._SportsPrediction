# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 18:42:09 2024

@author: user
"""

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Define the path to your .pkl file
model_path = r"C:\Users\user\Documents\streamlitdeploy\GradientBoosting (3).pkl"

# Load the model
try:
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    print("Model loaded successfully!")
    # Optionally, you can print or inspect the loaded model
    print("Loaded model:", model)

    # Function to preprocess input data
    def preprocess_input(preferred_foot, potential, age, shooting, passing, physic, movement_reactions):
        # One-hot encode the 'preferred_foot' field
        preferred_foot_left = 1 if preferred_foot == 'left' else 0
        preferred_foot_right = 1 if preferred_foot == 'right' else 0

        # Create a list with the input features in the required order
        input_data = [
            potential, age, shooting, passing, physic, movement_reactions,
            preferred_foot_left, preferred_foot_right
        ]

        return np.array(input_data).reshape(1, -1)  # Reshape to a 2D array

    # Function to handle prediction
    def predict_rating(preferred_foot, potential, age, shooting, passing, physic, movement_reactions):
        X_input = preprocess_input(preferred_foot, potential, age, shooting, passing, physic, movement_reactions)
        prediction = model.predict(X_input)[0]
        return prediction

    # Define routes
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        # Get data from POST request
        data = request.form

        # Extract data from form
        preferred_foot = data['preferred_foot']
        potential = float(data['potential'])
        age = float(data['age'])
        shooting = float(data['shooting'])
        passing = float(data['passing'])
        physic = float(data['physic'])
        movement_reactions = float(data['movement_reactions'])

        # Perform prediction
        prediction = predict_rating(preferred_foot, potential, age, shooting, passing, physic, movement_reactions)

        # Prepare response
        return render_template('result.html', prediction=prediction)

except FileNotFoundError:
    print(f"Error: The file '{model_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
