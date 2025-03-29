import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load Data
def load_data():
    df = pd.read_csv("Airport_Flight_Data_Final_Updated.csv")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df

# Preprocess Data
def preprocess_data(df):
    # Selecting relevant features for ML model
    features = ["Temperature", "Total_Flights", "Domestic_Flights", "International_Flights", "Load_Factor (%)", "Weather_Good", "Economic_Trend", "Is_Weekend", "Peak_Season", "Holiday"]
    target = "Actual_Footfall"
    
    df = df.dropna()  # Handle missing values
    X = df[features]
    y = df[target]
    return X, y, df

# Train Model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Model Performance
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    return model, mae, rmse

# Predict Future Footfall
def predict_future(model, df):
    future_X = df[["Temperature", "Total_Flights", "Domestic_Flights", "International_Flights", "Load_Factor (%)", "Weather_Good", "Economic_Trend", "Is_Weekend", "Peak_Season", "Holiday"]]
    df["Predicted_Footfall"] = model.predict(future_X)
    return df

# Streamlit App
st.title("Airport Footfall Prediction Dashboard ✈️")
df = load_data()
X, y, df = preprocess_data(df)
model, mae, rmse = train_model(X, y)
df = predict_future(model, df)

# Visualization
st.subheader("📊 Predicted vs Actual Footfall")
plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["Actual_Footfall"], label="Actual Footfall", marker="o", linestyle="-")
plt.plot(df["Date"], df["Predicted_Footfall"], label="Predicted Footfall", marker="x", linestyle="-")
plt.xlabel("Date")
plt.ylabel("Footfall Count")
plt.legend()
st.pyplot(plt)

# Display Metrics
st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"Root Mean Squared Error: {rmse:.2f}")

# Show Data
st.subheader("📂 Preview Data")
st.write(df.head())
