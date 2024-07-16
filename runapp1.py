import streamlit as st
import joblib
import numpy as np
# Load the model and label binarizer
model = joblib.load('genre_model.pkl')
mlb = joblib.load('mlb.pkl')
# Streamlit interface
st.title("Movie Genre Prediction App")
# Input movie description
description = st.text_area("Enter movie description:")

if st.button("Predict Genre"):
    if description:
        # Predict the genre
        pred = model.predict([description])
        pred_genres = mlb.inverse_transform(pred)
        
        # Display the results
        st.write("Predicted Genres:")
        for genre in pred_genres[0]:
            st.write(f"- {genre}")
    else:
        st.write("Please enter a movie description.")
