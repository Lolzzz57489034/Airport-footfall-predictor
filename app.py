import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("C:/Users/Dhruv Patel/Downloads/Airport_Flight_Data_Final_Updated.csv")

df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
df["Year"] = df["Date"].dt.year.fillna(df["Year"].mode()[0]).astype(int)

airports = [
    "Rajiv Gandhi International Airport",
    "Chhatrapati Shivaji Maharaj International Airport",
    "Indira Gandhi International Airport",
    "Kempegowda International Airport",
    "Sardar Vallabhbhai Patel International Airport",
    "Chennai International Airport"
]

seasons = ["Monsoon", "Summer", "Winter"]
flight_types = ["Domestic", "International"]

# Label Encoding for categorical values
le_airport = LabelEncoder()
df["Airport"] = le_airport.fit_transform(df["Airport"])
le_season = LabelEncoder()
df["Season"] = le_season.fit_transform(df["Season"])
le_flight = LabelEncoder()
df["Flight_Type"] = le_flight.fit_transform(df["Flight_Type"])

# Train a simple RandomForest model for prediction
features = ["Airport", "Season", "Flight_Type", "Year"]
X = df[features]
y = df["Footfall"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("✈ Airport Footfall Prediction")
st.subheader("🔮 Predict Future Airport Footfall")

# Select Airport
airport_choice = st.selectbox("Select an Airport:", airports)

# Convert airport to encoded value
if airport_choice in le_airport.classes_:
    airport_encoded = le_airport.transform([airport_choice])[0]
else:
    st.error("Selected airport not found in dataset!")
    st.stop()

# Select Season
season_choice = st.selectbox("Select a Season:", seasons)
season_encoded = le_season.transform([season_choice])[0]

# Select Flight Type
flight_choice = st.selectbox("Select Flight Type:", flight_types)
flight_encoded = le_flight.transform([flight_choice])[0]

# Select Future Year
future_year = st.slider("Select Future Year:", min_value=int(df["Year"].max() + 1), max_value=int(df["Year"].max() + 10), step=1)

# Make Prediction
if st.button("🔍 Predict"):
    input_data = np.array([[airport_encoded, season_encoded, flight_encoded, future_year]])
    prediction = model.predict(input_data)[0]
    
    st.success(f"Predicted Footfall for {airport_choice} in {future_year}: {int(prediction)}")
    
    # Visualization
    future_years = np.arange(df["Year"].max() + 1, df["Year"].max() + 11)
    predictions = [model.predict(np.array([[airport_encoded, season_encoded, flight_encoded, year]]))[0] for year in future_years]
    
    fig, ax = plt.subplots()
    ax.plot(future_years, predictions, marker='o', linestyle='-', color='b', label='Predicted Footfall')
    ax.set_xlabel("Year")
    ax.set_ylabel("Footfall")
    ax.set_title(f"Predicted Footfall Trend for {airport_choice}")
    ax.legend()
    st.pyplot(fig)
