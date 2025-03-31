import streamlit as st
import pandas as pd

# Load dataset
file_path = "Airport_Flight_Data_Final_Updated.csv"  # Ensure the correct path

@st.cache_data
def load_data():
    df = pd.read_csv(file_path)

# Display first few rows of the dataset
st.write("### Dataset Preview")
st.write(df.head())

# Check data types of columns
st.write("### Column Data Types")
st.write(df.dtypes)

