import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
file_path = "Airport_Flight_Data_Final_Updated.csv"  # Update if needed
df = pd.read_csv(file_path)

# Ensure required columns exist
required_columns = {"Year", "Predicted_Footfall", "Actual_Footfall"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    st.error(f"Missing columns: {missing_columns}. Please check the dataset.")
    st.stop()  # Stop execution if critical columns are missing

# Convert Year to numeric (if not already)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# Ensure no missing values in critical columns
df.dropna(subset=["Year", "Predicted_Footfall", "Actual_Footfall"], inplace=True)

# Feature Engineering
df["Footfall_Change"] = df["Actual_Footfall"].pct_change().fillna(0)

# Verify "Footfall_Change" exists before using it
if "Footfall_Change" not in df.columns:
    st.error("Footfall_Change column could not be computed. Check data consistency.")
    st.stop()

X = df[["Year", "Footfall_Change"]]
y = df["Actual_Footfall"]

# Train ML Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("Airport Footfall Prediction Dashboard")

tab1, tab2 = st.tabs(["📊 Historical Data", "🔮 Future Predictions"])

with tab1:
    st.subheader("Footfall Trends Over the Years")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=df["Year"], y=df["Actual_Footfall"], marker="o", label="Actual Footfall", ax=ax)
    sns.lineplot(x=df["Year"], y=df["Predicted_Footfall"], marker="o", label="Predicted Footfall", ax=ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("Footfall Count")
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.subheader("Future Footfall Prediction")
    future_year = st.slider("Select a Future Year", int(df["Year"].max()) + 1, int(df["Year"].max()) + 10)
    last_year_footfall = df.iloc[-1]["Actual_Footfall"]
    predicted_growth = df["Footfall_Change"].mean()
    predicted_footfall = last_year_footfall * (1 + predicted_growth) ** (future_year - df.iloc[-1]["Year"])
    
    st.write(f"**Predicted Footfall for {future_year}: {int(predicted_footfall)}**")
    
    # Plot past + future trends
    future_years = np.arange(df["Year"].max(), future_year + 1)
    future_predictions = [last_year_footfall * (1 + predicted_growth) ** (y - df.iloc[-1]["Year"]) for y in future_years]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=df["Year"], y=df["Actual_Footfall"], marker="o", label="Actual Footfall", ax=ax)
    sns.lineplot(x=future_years, y=future_predictions, marker="o", linestyle="dashed", label="Predicted Footfall", ax=ax)
    ax.axvline(x=future_year, color="red", linestyle="--", label="Prediction Point")
    st.pyplot(fig)
