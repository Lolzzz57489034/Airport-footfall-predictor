import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Airport_Flight_Data_Final_Updated.csv")
    
    # Ensure 'Date' column is in datetime format
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=["Date"])
    
    return df

df = load_data()

# UI - Title & Selections
st.title("✈️ Airport Footfall Prediction")
st.subheader("🔮 Predict Future Airport Footfall")

# Extract unique airport names
airports = df["Airport"].unique().tolist()

# Dropdown to select airport
selected_airport = st.selectbox("Select an Airport:", airports)

# Dropdown to select season
season = st.selectbox("Select Season:", ["Winter", "Summer", "Monsoon"])

# Dropdown for flight type
flight_type = st.selectbox("Select Flight Type:", ["Domestic", "International"])

# Select Future Year
future_year = st.slider("Select Future Year:", min_value=2025, max_value=2035, step=1)

# Filter data based on selection
filtered_df = df[(df["Airport"] == selected_airport) & (df["Season"] == season)]

if flight_type == "Domestic":
    target_column = "Domestic_Flights"
else:
    target_column = "International_Flights"

# Prepare data for ML model
if not filtered_df.empty:
    filtered_df["Year"] = filtered_df["Date"].dt.year.astype(int)
    X = filtered_df[["Year"]]
    y = filtered_df[target_column]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make prediction
    future_pred = model.predict([[future_year]])
    future_pred = max(0, future_pred[0])  # Ensure non-negative prediction
    
    st.success(f"Predicted {flight_type} flights for {selected_airport} in {future_year}: {int(future_pred)}")
    
    # Visualize
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Actual Data')
    ax.plot(X, model.predict(X), color='red', label='Prediction Trend')
    ax.scatter([future_year], [future_pred], color='green', label='Future Prediction', s=100)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Number of {flight_type} Flights")
    ax.legend()
    st.pyplot(fig)
else:
    st.error("No data available for the selected filters!")
