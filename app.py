import streamlit as st
import joblib
import numpy as np

# Load the saved logistic regression model
model = joblib.load("iris_logistic_regression_model.pkl")

st.title("🌼 Iris Flower Classification App")
st.write("This app predicts the type of Iris flower based on input measurements.")

# Input fields for features
sepal_length = st.number_input("Sepal Length (cm)", value=5.1, format="%.2f")
sepal_width = st.number_input("Sepal Width (cm)", value=3.5, format="%.2f")
petal_length = st.number_input("Petal Length (cm)", value=1.4, format="%.2f")
petal_width = st.number_input("Petal Width (cm)", value=0.2, format="%.2f")

# Convert inputs to numpy array
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Predict Flower Type"):
    prediction = model.predict(features)[0]
    flower_classes = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"🌱 Predicted Flower Type: **{flower_classes[prediction]}**")
