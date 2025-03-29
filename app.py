import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import requests
import io

# Load dataset from GitHub
url = "https://raw.githubusercontent.com/Lolzzz57489034/Airport-footfall-predictor/main/Airport_Flight_Data_Final_Updated.csv"
response = requests.get(url)
df = pd.read_csv(io.StringIO(response.text))

# Ensure 'Date' is in datetime format
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Extract 'Year' as integer
df["Year"] = df["Date"].dt.year.fillna(df["Date"].dt.year.mode()[0]).astype(int)

# Encode categorical features
label_encoders = {}
categorical_cols = ["Season", "Weather_Good", "Economic_Trend", "Airport"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store for future use

# Prepare dataset for ML
X = df[["Year", "Airport", "Season", "Weather_Good", "Economic_Trend", "Total_Flights"]]
y = df["Actual_Footfall"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ML Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title="Airport Footfall Predictor", layout="wide")
st.title("\U00002708 Airport Footfall Prediction")

# User Inputs
airport_list = df["Airport"].unique()
selected_airport = st.selectbox("Select Airport:", airport_list)

season_list = ["Monsoon", "Summer", "Winter"]
selected_season = st.selectbox("Select Season:", season_list)

flight_type = st.radio("Flight Type:", ["Domestic", "International"])

future_year = st.slider("Select Future Year:", min_value=df["Year"].max() + 1, max_value=df["Year"].max() + 10, step=1)

# Predict button
if st.button("\U0001F680 Predict"):
    # Encode input values
    if selected_airport in label_encoders["Airport"].classes_:
        airport_encoded = label_encoders["Airport"].transform([selected_airport])[0]
    else:
        st.error("❌ Selected airport not found in dataset!")
        st.stop()
    
    season_encoded = label_encoders["Season"].transform([selected_season])[0]
    flight_encoded = 1 if flight_type == "Domestic" else 0  # Assuming binary encoding
    
    # Prepare input for prediction
    input_data = np.array([[future_year, airport_encoded, season_encoded, 1, 1, 100]])
    predicted_footfall = model.predict(input_data)[0]

    # Display Prediction
    st.subheader(f"\U0001F4CA Predicted Footfall: **{int(predicted_footfall)} passengers**")

    # Visualization
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=df["Year"], y=df["Actual_Footfall"], marker="o", label="Past Data")
    plt.axvline(x=future_year, color="r", linestyle="--", label="Prediction Point")
    plt.scatter(future_year, predicted_footfall, color="red", s=100, label="Predicted Footfall")
    plt.xlabel("Year")
    plt.ylabel("Passenger Footfall")
    plt.legend()
    st.pyplot(plt)
