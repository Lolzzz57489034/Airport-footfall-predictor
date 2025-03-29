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

# Convert 'Date' to datetime and extract 'Year'
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year

# Define airport list dynamically from dataset
airport_list = df["Airport"].unique().tolist()

# Streamlit UI
st.set_page_config(page_title="Airport Footfall Predictor", layout="wide")
st.title("\U00002708 Airport Footfall Prediction")

# User selects an airport
selected_airport = st.selectbox("Select an Airport:", airport_list)

# Filter dataset based on selected airport
df_airport = df[df["Airport"] == selected_airport]

# User selects a season
season_list = ["Winter", "Monsoon", "Summer"]
selected_season = st.selectbox("Select a Season:", season_list)

# User selects flight type
flight_type = st.radio("Select Flight Type:", ["Domestic", "International"])

# Extract relevant features
X = df[["Year", "Season", "Total_Flights", "Domestic_Flights", "International_Flights", "Load_Factor (%)", "Economic_Trend"]]
y = df["Actual_Footfall"]

# Encode categorical features
label_encoders = {}
for col in ["Season"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Train ML Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict future footfall
future_year = st.slider("Select Future Year:", min_value=df["Year"].max() + 1, max_value=df["Year"].max() + 10, step=1)

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
