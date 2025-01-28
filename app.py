
import streamlit as st
import pickle
import numpy as np

# Load the pre-trained SVM model
with open('svm_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Backend function to make predictions
def predict(input_features):
    return model.predict(np.array([input_features]))[0]

# Streamlit App Title
st.title("SVM Prediction Web App")
st.write("Use this application to predict the class based on two input features using a pre-trained SVM model.")

# Sidebar for instructions
st.sidebar.title("Instructions")
st.sidebar.write("""
1. Enter values for **Feature 1** and **Feature 2** using the sliders or input boxes.
2. Click **Predict** to see the class prediction.
3. The result will appear below the button.
""")

# Main form for user input
st.header("Enter Input Features")
feature1 = st.slider("Feature 1 (x1):", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
feature2 = st.slider("Feature 2 (x2):", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)

# Predict button
if st.button("Predict"):
    # Make prediction using the backend function
    result = predict([feature1, feature2])
    # Display the result
    st.success(f"Prediction: **Class {result}**")
    if result == 1:
        st.info("This indicates the data point belongs to Class 1.")
    else:
        st.info("This indicates the data point belongs to Class 0.")
