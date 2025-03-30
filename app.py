import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset from GitHub repository
github_url = "https://raw.githubusercontent.com/Lolzzz57489034/Airport-footfall-predictor/main/Airport_Flight_Data_Final_Updated.csv"
df = pd.read_csv(github_url)

# Ensure 'Date' is in datetime format
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year.fillna(df["Date"].dt.year.mode()[0]).astype(int)

# Define airport names
airport_names = [
    "Rajiv Gandhi International Airport",
    "Chhatrapati Shivaji Maharaj International Airport",
    "Indira Gandhi International Airport",
    "Kempegowda International Airport",
    "Sardar Vallabhbhai Patel International Airport",
    "Chennai International Airport"
]

# Encode categorical features
label_encoders = {}
categorical_cols = ["Season", "Weather_Good", "Economic_Trend", "Airport"]
for col in categorical_cols:
    df[col] = df[col].astype(str)  # Ensure categorical data is string
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Create a synthetic revenue column
df["Revenue"] = df["Actual_Footfall"] * np.random.uniform(20, 50, size=len(df))

# Prepare dataset for ML
X_footfall = df[["Year", "Airport", "Season", "Weather_Good", "Economic_Trend", "Total_Flights"]]
y_footfall = df["Actual_Footfall"]
X_train_footfall, X_test_footfall, y_train_footfall, y_test_footfall = train_test_split(X_footfall, y_footfall, test_size=0.2, random_state=42)

# Train Footfall Prediction Model
footfall_model = RandomForestRegressor(n_estimators=100, random_state=42)
footfall_model.fit(X_train_footfall, y_train_footfall)

# Streamlit UI
st.set_page_config(page_title="Airport Footfall & Revenue Predictor", layout="wide")
st.title("\U0001F6EB Airport Footfall & Revenue Prediction")

# Dropdown for airport selection
selected_airport = st.selectbox("Select Airport:", airport_names)

# Map selected airport to encoded value
selected_airport_encoded = label_encoders["Airport"].transform([selected_airport])[0]

# Dropdown for season selection
seasons = list(label_encoders["Season"].classes_)
selected_season = st.selectbox("Select Season:", seasons)
selected_season_encoded = label_encoders["Season"].transform([selected_season])[0]

# Predict Footfall for Next 10 Years
future_years = list(range(df["Year"].max() + 1, df["Year"].max() + 11))
predicted_footfall_values = []
for year in future_years:
    input_data = np.array([[year, selected_airport_encoded, selected_season_encoded, 1, 1, 1000]])
    predicted_footfall = footfall_model.predict(input_data.reshape(1, -1))[0]
    predicted_footfall_values.append(predicted_footfall)

# Display Footfall Prediction Results
st.subheader("\U0001F4CA Predicted Footfall for Next 10 Years")
predicted_footfall_df = pd.DataFrame({"Year": future_years, "Predicted Footfall": predicted_footfall_values})
st.dataframe(predicted_footfall_df)

# Visualization for Footfall Prediction
plt.figure(figsize=(8, 5))
sns.lineplot(x=predicted_footfall_df["Year"], y=predicted_footfall_df["Predicted Footfall"], marker="o", label="Predicted Footfall")
plt.xlabel("Year")
plt.ylabel("Passenger Footfall")
plt.legend()
st.pyplot(plt)

# Option to Predict Revenue
if st.checkbox("Predict Revenue Based on Footfall"):
    # Train Revenue Prediction Model
    X_revenue = df[["Actual_Footfall", "Total_Flights"]]
    y_revenue = df["Revenue"]
    X_train_revenue, X_test_revenue, y_train_revenue, y_test_revenue = train_test_split(X_revenue, y_revenue, test_size=0.2, random_state=42)
    
    revenue_model = RandomForestRegressor(n_estimators=100, random_state=42)
    revenue_model.fit(X_train_revenue, y_train_revenue)
    
    # Predict Revenue
    predicted_revenue_values = revenue_model.predict(np.array(predicted_footfall_values).reshape(-1, 1))
    
    # Display Revenue Prediction Results
    st.subheader("\U0001F4B0 Predicted Revenue for Next 10 Years")
    predicted_revenue_df = pd.DataFrame({"Year": future_years, "Predicted Revenue ($)": predicted_revenue_values})
    st.dataframe(predicted_revenue_df)
    
    # Visualization for Revenue Prediction
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=predicted_revenue_df["Year"], y=predicted_revenue_df["Predicted Revenue ($)"], marker="o", label="Predicted Revenue")
    plt.xlabel("Year")
    plt.ylabel("Revenue ($)")
    plt.legend()
    st.pyplot(plt)
