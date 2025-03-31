import streamlit as st
import pandas as pd

# Load dataset
file_path = "Airport_Flight_Data_Final_Updated.csv"  # Ensure the correct path

@st.cache_data
def load_data():
    df = pd.read_csv(file_path)

    # Check if 'Date' column exists
    if "Date" not in df.columns:
        st.error("Error: 'Date' column not found in the dataset.")
        return None

    # Print out first few values
    st.write("### Sample values from 'Date' column:")
    st.write(df["Date"].head(10))

    return df

