import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import random

# Load dataset
file_path = "Airport_Flight_Data_Final_Updated.csv"
df = pd.read_csv(file_path)

# Ensure 'Date' column exists and is in datetime format
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df.dropna(subset=["Date"], inplace=True)
df["Year"] = df["Date"].dt.year.astype(int)

# Filter dataset based on provided airports
airports = [
    "Rajiv Gandhi International Airport", 
    "Chhatrapati Shivaji Maharaj International Airport", 
    "Indira Gandhi International Airport", 
    "Kempegowda International Airport", 
    "Sardar Vallabhbhai Patel International Airport", 
    "Chennai International Airport"
]
df = df[df["Airport"].isin(airports)]

# Streamlit UI
st.set_page_config(page_title="Airport Footfall Predictor", layout="wide")
st.title("\U00002708 Airport Footfall Prediction")

# User selects an airport
selected_airport = st.selectbox("Select an Airport:", airports)

# Filter dataset based on selected airport
df_airport = df[df["Airport"] == selected_airport]

# User selects a season
seasons = ["Monsoon", "Summer", "Winter"]
selected_season = st.selectbox("Select a Season:", seasons)

# User selects flight type
flight_type = st.radio("Select Flight Type:", ["Domestic", "International"])

# Extract relevant features
features = ["Year", "Season", "Total_Flights", "Domestic_Flights", "International_Flights", "Load_Factor (%)", "Economic_Trend"]
X = df[features]
y = df["Actual_Footfall"]

# Encode categorical features
label_encoders = {}
for col in ["Season"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Train ML Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Ensure max_year is valid
max_year = int(df["Year"].max())
future_year = st.slider("Select Future Year:", min_value=max_year + 1, max_value=max_year + 10, step=1)

# Prepare input for prediction
season_encoded = label_encoders["Season"].transform([selected_season])[0]
if flight_type == "Domestic":
    domestic_flights = int(df_airport["Domestic_Flights"].mean())
    international_flights = 0
else:
    domestic_flights = 0
    international_flights = int(df_airport["International_Flights"].mean())

total_flights = domestic_flights + international_flights
load_factor = df_airport["Load_Factor (%)"].mean()
economic_trend = df_airport["Economic_Trend"].mean()

input_data = np.array([[future_year, season_encoded, total_flights, domestic_flights, international_flights, load_factor, economic_trend]])
predicted_footfall = model.predict(input_data)[0]

# Display Prediction
st.subheader(f"\U0001F4CA Predicted Footfall: **{int(predicted_footfall)} passengers**")

# Visualization
plt.figure(figsize=(8, 5))
sns.lineplot(x=df_airport["Year"], y=df_airport["Actual_Footfall"], marker="o", label="Past Data")
plt.axvline(x=future_year, color="r", linestyle="--", label="Prediction Point")
plt.scatter(future_year, predicted_footfall, color="red", s=100, label="Predicted Footfall")
plt.xlabel("Year")
plt.ylabel("Passenger Footfall")
plt.legend()
st.pyplot(plt)
