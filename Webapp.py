import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open('lung_cancer_model.sav', 'rb'))

# Create a function for prediction
def Lung_cancer_prediction(input_data):
    # Convert input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Debug: Print input data
    st.write("Input data for prediction:", input_data_reshaped)

    # Make prediction
    prediction = loaded_model.predict(input_data_reshaped)

    # Debug: Print prediction output
    st.write("Raw prediction result:", prediction)

    if prediction[0] == 0:
        return 'The person does not have lung cancer'
    else:
        return 'The person has lung cancer'

def main():
    # App title
    st.title('Lung Cancer Prediction Web App')

    # Getting input data from the user as number and selection inputs
    gender = st.selectbox("Gender", ["M", "F"])
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    smoking = st.selectbox("Smoking Habit", ["Yes", "No"])
    yellow_fingers = st.selectbox("Yellow Fingers", ["Yes", "No"])
    anxiety = st.selectbox("Anxiety", ["Yes", "No"])
    peer_pressure = st.selectbox("Peer Pressure", ["Yes", "No"])
    chronic_disease = st.selectbox("Chronic Disease", ["Yes", "No"])
    fatigue = st.selectbox("Fatigue", ["Yes", "No"])
    allergy = st.selectbox("Allergy", ["Yes", "No"])
    wheezing = st.selectbox("Wheezing", ["Yes", "No"])
    alcohol_consuming = st.selectbox("Alcohol Consuming", ["Yes", "No"])
    coughing = st.selectbox("Coughing", ["Yes", "No"])
    shortness_of_breath = st.selectbox("Shortness of Breath", ["Yes", "No"])
    swallowing_difficulty = st.selectbox("Swallowing Difficulty", ["Yes", "No"])
    chest_pain = st.selectbox("Chest Pain", ["Yes", "No"])

    # Convert binary fields to integers (Yes=2, No=1)
    binary_map = {"Yes": 2, "No": 1}
    diagnosis = ''

    if st.button('Lung Cancer Result'):
        try:
            # Convert selections to required numerical format
            diagnosis = Lung_cancer_prediction([
                1 if gender == "M" else 0,  # Assuming M/F is represented as 1/0
                age,
                binary_map[smoking], binary_map[yellow_fingers], binary_map[anxiety],
                binary_map[peer_pressure], binary_map[chronic_disease], binary_map[fatigue],
                binary_map[allergy], binary_map[wheezing], binary_map[alcohol_consuming],
                binary_map[coughing], binary_map[shortness_of_breath],
                binary_map[swallowing_difficulty], binary_map[chest_pain]
            ])
            st.success(diagnosis)
        except ValueError:
            st.error("Please enter valid numeric values for age and binary choices.")

if __name__ == '__main__':
    main()
