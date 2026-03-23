import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Breast Cancer Prediction")

radius = st.number_input("Radius Mean")
texture = st.number_input("Texture Mean")

if st.button("Predict"):
    data = np.array([[radius, texture]])
    data = scaler.transform(data)
    pred = model.predict(data)

    if pred[0] == "M":
        st.error("Malignant")
    else:
        st.success("Benign")
