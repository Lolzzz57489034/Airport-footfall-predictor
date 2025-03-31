@st.cache_data
def load_data():
    df = pd.read_csv(file_path)

    if "Date" not in df.columns:
        st.error("Error: 'Date' column not found in the dataset.")
        st.stop()

    st.write("### First few values in 'Date' column:")
    st.write(df["Date"].head(10))

    st.write(f"Data type of 'Date' column: {df['Date'].dtype}")

    df["Date"] = pd.to_datetime(df["Date"].str.strip(), errors="coerce")

    st.write("### Rows where Date conversion failed:")
    st.write(df[df["Date"].isna()].head(10))

    df = df.dropna(subset=["Date"])  

    return df
