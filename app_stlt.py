import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

# ----------------------
# Load model & scaler
# ----------------------
MODEL_PATH = r"C:\Python\Project\seoul bike\xgboost_regressor_adj_r2_0.954_v1.pkl"
SCALER_PATH = r"C:\Python\Project\seoul bike\scalar.pkl"

model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))

# ----------------------
# Helpers
# ----------------------
def add_date_features(data):
    day = int(data["Day"])
    month = int(data["Month"])
    year = int(data["Year"])
    dt = datetime(year, month, day)
    week_day = dt.strftime("%A")
    for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        data[d] = 1 if d == week_day else 0
    return data

def preprocess_input(data):
    df = pd.DataFrame([data])
    df = add_date_features(df.iloc[0].to_dict())
    df = pd.DataFrame([df])
    expected_features = scaler.feature_names_in_
    df = df.reindex(columns=expected_features, fill_value=0)
    return scaler.transform(df)

def make_prediction(data_dict):
    scaled = preprocess_input(data_dict)
    pred = np.square(model.predict(scaled))
    return int(round(pred[0]))

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Seoul Bike Prediction", page_icon="🚲", layout="wide")
st.title("🚲 Seoul Bike Rental Prediction")

st.markdown("Enter conditions below to estimate the number of rented bikes in Seoul.")

# ----------------------
# Input columns
# ----------------------
col1, col2, col3 = st.columns(3)

# ----------------------
# Input columns with headings
# ----------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📅 Date & Time")
    date = st.date_input("Date", format="DD/MM/YYYY")
    hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, step=1)
    season = st.selectbox("Season", ["Autumn", "Spring", "Summer", "Winter"])

with col2:
    st.markdown("### 🌤 Weather Conditions")
    temperature = st.number_input("Temperature (°C)", value=20.0)
    humidity = st.number_input("Humidity (%)", value=50.0)
    wind_speed = st.number_input("Wind Speed (m/s)", value=1.5)

with col3:
    st.markdown("### ⚡ Other Factors")
    visibility = st.number_input("Visibility (10m)", value=2000.0)
    solar_radiation = st.number_input("Solar Radiation (MJ/m2)", value=0.0)
    rainfall = st.number_input("Rainfall (mm)", value=0.0)
    snowfall = st.number_input("Snowfall (cm)", value=0.0)
    holiday = st.selectbox("Holiday", ["No Holiday", "Holiday"])
    functioning_day = st.selectbox("Functioning Day", ["Yes", "No"])


# ----------------------
# Prediction Button (Centered)
# ----------------------
st.markdown("---")
center_col = st.columns([1, 2, 1])[1]  # wider middle column
with center_col:
    predict_btn = st.button("Predict Bike Rentals", use_container_width=True)

# ----------------------
# Prediction Result
# ----------------------
if predict_btn:
    data = {
        "Hour": hour,
        "Temperature(°C)": temperature,
        "Humidity(%)": humidity,
        "Wind speed (m/s)": wind_speed,
        "Visibility (10m)": visibility,
        "Solar Radiation (MJ/m2)": solar_radiation,
        "Rainfall(mm)": rainfall,
        "Snowfall (cm)": snowfall,
        "Holiday": 1 if holiday == "Holiday" else 0,
        "Functioning Day": 1 if functioning_day == "Yes" else 0,
        "Day": date.day,
        "Month": date.month,
        "Year": date.year,
        "Autumn": 1 if season == "Autumn" else 0,
        "Spring": 1 if season == "Spring" else 0,
        "Summer": 1 if season == "Summer" else 0,
        "Winter": 1 if season == "Winter" else 0,
    }

    scaled = preprocess_input(data)
    pred = np.square(model.predict(scaled))
    pred_value = int(round(pred[0]))

    # Centered and big prediction
    st.markdown("### 🎯 Prediction Result")
    result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
    with result_col2:
        st.markdown(
            f"<h1 style='text-align: center; color: green; font-size: 60px;'>🚲 {pred_value:,}</h1>", 
            unsafe_allow_html=True
        )
        st.markdown("<p style='text-align: center; font-size:20px;'>Predicted Rented Bike Count</p>", unsafe_allow_html=True)

