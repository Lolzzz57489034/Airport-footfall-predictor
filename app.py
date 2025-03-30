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
airport_names = df["Airport"].unique().tolist()

# Encode categorical features
label_encoders = {}
categorical_cols = ["Season", "Weather_Good", "Economic_Trend", "Airport"]
for col in categorical_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Create a revenue formula: Revenue = Footfall * Avg Spending per Passenger (INR)
avg_spending_per_passenger = 2500  # Assumption: 2500 INR per passenger
df["Revenue"] = df["Actual_Footfall"] * avg_spending_per_passenger

# Train Footfall Prediction Model
X_footfall = df[["Year", "Airport", "Season", "Weather_Good", "Economic_Trend", "Total_Flights"]]
y_footfall = df["Actual_Footfall"]
X_train_footfall, X_test_footfall, y_train_footfall, y_test_footfall = train_test_split(X_footfall, y_footfall, test_size=0.2, random_state=42)
footfall_model = RandomForestRegressor(n_estimators=100, random_state=42)
footfall_model.fit(X_train_footfall, y_train_footfall)

# Train Revenue Prediction Model
X_revenue = df[["Actual_Footfall", "Total_Flights"]]
y_revenue = df["Revenue"]
X_train_revenue, X_test_revenue, y_train_revenue, y_test_revenue = train_test_split(X_revenue, y_revenue, test_size=0.2, random_state=42)
revenue_model = RandomForestRegressor(n_estimators=100, random_state=42)
revenue_model.fit(X_train_revenue, y_train_revenue)

# Streamlit UI
st.set_page_config(page_title="Airport Footfall & Revenue Predictor", layout="wide")
st.title("\U0001F6EB Airport Footfall & Revenue Prediction")

# User Input Section
selected_airport = st.selectbox("Select Airport:", airport_names)
selected_season = st.selectbox("Select Season:", list(label_encoders["Season"].classes_))
selected_year = st.slider("Select Year:", min_value=df["Year"].max() + 1, max_value=df["Year"].max() + 10, step=1)

# Encode user selections
selected_airport_encoded = label_encoders["Airport"].transform([selected_airport])[0]
selected_season_encoded = label_encoders["Season"].transform([selected_season])[0]

# Predict Footfall
input_data = np.array([[selected_year, selected_airport_encoded, selected_season_encoded, 1, 1, 1000]])
predicted_footfall = footfall_model.predict(input_data.reshape(1, -1))[0]

st.subheader(f"\U0001F4CA Predicted Footfall for {selected_year}: {int(predicted_footfall):,} Passengers")

# Predict Revenue
predicted_revenue = revenue_model.predict(np.array([[predicted_footfall, 1000]]))[0]

st.subheader(f"\U0001F4B0 Predicted Revenue for {selected_year}: ₹{int(predicted_revenue):,}")

# Visualization
future_years = list(range(df["Year"].max() + 1, df["Year"].max() + 11))
predicted_footfall_values = []
predicted_revenue_values = []
for year in future_years:
    input_data = np.array([[year, selected_airport_encoded, selected_season_encoded, 1, 1, 1000]])
    footfall = footfall_model.predict(input_data.reshape(1, -1))[0]
    revenue = revenue_model.predict(np.array([[footfall, 1000]]))[0]
    predicted_footfall_values.append(footfall)
    predicted_revenue_values.append(revenue)

# Plot Footfall Prediction
plt.figure(figsize=(8, 5))
sns.lineplot(x=future_years, y=predicted_footfall_values, marker="o", label="Predicted Footfall")
plt.xlabel("Year")
plt.ylabel("Passenger Footfall")
plt.legend()
st.pyplot(plt)

# Plot Revenue Prediction
plt.figure(figsize=(8, 5))
sns.lineplot(x=future_years, y=predicted_revenue_values, marker="o", label="Predicted Revenue (INR)")
plt.xlabel("Year")
plt.ylabel("Revenue (INR)")
plt.legend()
st.pyplot(plt)
