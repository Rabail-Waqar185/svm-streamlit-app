
import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

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
st.sidebar.info(
    """
    1. Use the sliders to input values for **Feature 1** and **Feature 2**.
    2. Click the **Predict** button to get the class prediction.
    3. View a real-time scatter plot of your input compared to the dataset.
    """
)

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

    # Visualize the prediction
    st.subheader("Scatter Plot with Your Input")
    fig, ax = plt.subplots()

    # Generate example dataset for visualization
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=400, n_features=2, centers=2, cluster_std=1.2, random_state=0)

    # Plot dataset points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', alpha=0.6, label='Dataset')

    # Highlight user input
    ax.scatter(feature1, feature2, color='red', label='Your Input', s=100, edgecolors='black')
    ax.set_xlabel("Feature 1 (x1)")
    ax.set_ylabel("Feature 2 (x2)")
    ax.legend()

    # Show the plot
    st.pyplot(fig)
