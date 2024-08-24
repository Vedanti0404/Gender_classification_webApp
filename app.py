import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the model (ensure this is done before calling the function)
with open('D:/languages/python/machine_learning/Notes/diabetes_ML_WebApp/Random_Forest_trained_model.sav', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

def gender_prediction(input_data):
    """
    Predict the gender based on input features.
    
    Parameters:
    input_data (list or array-like): List or array-like object containing feature values.
    
    Returns:
    str: Prediction result.
    """
    
    # Ensure input_data is a list or array-like
    if not isinstance(input_data, (list, np.ndarray)):
        raise ValueError("Input data should be a list or a numpy array")

    # Convert the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Check that the number of features matches the model's expected input
    if input_data_as_numpy_array.shape[0] != 7:
        raise ValueError("Input data must have 7 features")

    # Reshape the array to be 2D (1 instance, 7 features)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make a prediction
    prediction = loaded_model.predict(input_data_reshaped)

    # Interpret the prediction
    if prediction[0] == 0:
        return 'The person is Female'
    else:
        return 'The person is Male'

def main():
    # Giving a title
    st.title('Gender Prediction Web App')

    # Text input fields
    long_hair = st.text_input('Do you have long hair? (1 for Yes, 0 for No)')
    forehead_width_cm = st.text_input('Forehead width in cm')
    forehead_height_cm = st.text_input('Forehead height in cm')
    nose_wide = st.text_input('Nose wide (1 for Yes, 0 for No)')
    nose_long = st.text_input('Nose long (1 for Yes, 0 for No)')
    lips_thin = st.text_input('Lips thin (1 for Yes, 0 for No)')
    distance_nose_to_lip_long = st.text_input('Distance from nose to lip long (1 for Yes, 0 for No)')

    # Convert inputs to floats
    try:
        input_data = [
            float(long_hair),
            float(forehead_width_cm),
            float(forehead_height_cm),
            float(nose_wide),
            float(nose_long),
            float(lips_thin),
            float(distance_nose_to_lip_long)
        ]
    except ValueError:
        st.error("Please enter valid numeric values.")
        return

    # Code for Prediction
    diagnosis = ''
    
    # Creating a button for Prediction
    if st.button('Gender Prediction Result'):
        try:
            diagnosis = gender_prediction(input_data)
            st.success(diagnosis)
        except ValueError as e:
            st.error(f"Error: {e}")

if __name__ == '__main__':
    main()
