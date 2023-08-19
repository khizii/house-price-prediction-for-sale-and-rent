import streamlit as st
import pandas as pd
import joblib
import nltk
from nltk.stem import PorterStemmer
nltk.download('punkt')

stemmer = PorterStemmer()

# Load trained models
sale_model = joblib.load('C:\\Users\\USER\\Desktop\\house-price-prediction\\trained_sales_data.pkl')
rent_model = joblib.load('C:\\Users\\USER\\Desktop\\house-price-prediction\\trained_rent_data.pkl')

# Streamlit UI
st.title("Soltridge house Price Prediction System")
choice = st.selectbox("Choose Listing Type", ['For Sale', 'For Rent'])

property_type = st.text_input("Property Type")
city = st.text_input("City")
location = st.text_input("Location")
baths = st.number_input("Number of Baths", value=1.0)
bedrooms = st.number_input("Number of Bedrooms", value=1.0)
total_area = st.number_input("Total Area")

stemmed_property_type = ' '.join([stemmer.stem(word) for word in nltk.word_tokenize(property_type)]).lower()
stemmed_city_type = ' '.join([stemmer.stem(word) for word in nltk.word_tokenize(city)]).lower()
stemmed_location_type = ' '.join([stemmer.stem(word) for word in nltk.word_tokenize(location)]).lower()

# Preprocess input
input_data = pd.DataFrame({
    'property_type': [stemmed_property_type],
    'baths': [baths],
    'bedrooms': [bedrooms],
    'location': [stemmed_location_type],
    'city': [stemmed_city_type],
    'Total_Area': [total_area]
})

# Make prediction
if choice == 'For Sale':
    selected_model = sale_model
    model_type = 'Sale'
else:
    selected_model = rent_model
    model_type = 'Rent'

if st.button("Predict"):
    prediction = selected_model.predict(input_data)
    st.success(f"Predicted {model_type} Price: {prediction[0]:.2f} Rs")
