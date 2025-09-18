# REALTIME DROUGHT ANOMALY DETECTION
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta

# -------------------- Load Config --------------------
load_dotenv()

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Real-Time Drought Anomaly Detection", layout="wide")
st.title("🌾 Real-Time Drought Anomaly Detection for Early Warning (Hyderabad)")

# Auto-refresh every 5 minutes
st_autorefresh(interval=5 * 60 * 1000, key="refresh", limit=100)
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Sidebar
st.sidebar.header("🔍 Filters")
variables = ["temperature", "precipitation", "soil_moisture", "rainfall_deficit_pct"]
selected_variable = st.sidebar.selectbox("Select variable", variables)

st.sidebar.header("⚙️ Anomaly Settings")
contamination = st.sidebar.slider("Contamination rate", 0.01, 0.1, 0.03, 0.01)

# -------------------- Data Fetching --------------------
@st.cache_data(ttl=3600)
def fetch_live_data():
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 17.3850,
            "longitude": 78.4867,
            "hourly": "temperature_2m,precipitation,soil_moisture_0_1cm",
            "timezone": "Asia/Kolkata",
            "past_days": 90
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame({
            "timestamp": data["hourly"]["time"],
            "temperature": data["hourly"]["temperature_2m"],
            "precipitation": data["hourly"]["precipitation"],
            "soil_moisture": data["hourly"]["soil_moisture_0_1cm"]
        })

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.dropna()
        return df

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def check_data_completeness(df):
    required = ["temperature", "precipitation", "soil_moisture"]
    missing = [col for col in required if col not in df.columns or df[col].isnull().all()]
    if missing:
        st.warning(f"Missing or empty columns: {missing}")

def calculate_rainfall_deficit(df, expected_daily_rain=5.0):
    df_daily = df.resample('D', on='timestamp').precipitation.sum().reset_index()
    df_daily["expected_rainfall"] = expected_daily_rain
    df_daily["rolling_precipitation"] = df_daily["precipitation"].rolling(7, min_periods=1).sum()
    df_daily["rolling_expected"] = df_daily["expected_rainfall"].rolling(7, min_periods=1).sum()
    df_daily["rainfall_deficit_pct"] = (
        100 * (df_daily["rolling_expected"] - df_daily["rolling_precipitation"]) / df_daily["rolling_expected"]
    )
    df_daily["rainfall_deficit_pct"] = df_daily["rainfall_deficit_pct"].clip(lower=0)
    return df_daily[["timestamp", "rainfall_deficit_pct"]]

# -------------------- Anomaly Detection --------------------
@st.cache_data(ttl=3600)
def detect_anomalies_multivariate(df, rainfall_deficit_df, contamination=0.03):
    df = df.copy()
    df["date"] = df["timestamp"].dt.floor('D')

    # Rolling features
    df["temp_roll_mean_24h"] = df["temperature"].rolling(24, min_periods=1).mean()
    df["temp_roll_mean_72h"] = df["temperature"].rolling(72, min_periods=1).mean()
    df["precip_roll_mean_24h"] = df["precipitation"].rolling(24, min_periods=1).mean()
    df["precip_roll_mean_72h"] = df["precipitation"].rolling(72, min_periods=1).mean()
    df["soil_moist_roll_mean_24h"] = df["soil_moisture"].rolling(24, min_periods=1).mean()
    df["soil_moist_roll_mean_72h"] = df["soil_moisture"].rolling(72, min_periods=1).mean()

    # Merge rainfall deficit
    rainfall_deficit_df["date"] = rainfall_deficit_df["timestamp"].dt.floor('D')
    df = pd.merge(df, rainfall_deficit_df[["date", "rainfall_deficit_pct"]], on="date", how="left")

    # Features
    features = df[[
        "temperature", "precipitation", "soil_moisture",
        "temp_roll_mean_24h", "temp_roll_mean_72h",
        "precip_roll_mean_24h", "precip_roll_mean_72h",
        "soil_moist_roll_mean_24h", "soil_moist_roll_mean_72h",
        "rainfall_deficit_pct"
    ]].fillna(method="ffill").fillna(method="bfill")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', IsolationForest(contamination=contamination, random_state=42))
    ])
    pipeline.fit(features)

    df["multivariate_anomaly"] = (pipeline.predict(features) == -1).astype(int)
    anomaly_ratio = df["multivariate_anomaly"].mean()  # fraction of anomalies
    return df, anomaly_ratio

# -------------------- Alerts --------------------
def send_email_alert(subject, body):
    if not all([SENDER_EMAIL, RECEIVER_EMAIL, EMAIL_PASSWORD]):
        st.warning("Missing email configuration.")
        return

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        st.success("📧 Email alert sent successfully.")
    except Exception as e:
        st.error(f"Email send error: {e}")

# -------------------- Visualizations --------------------
def plot_anomalies(df, variable):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["timestamp"], df[variable], label=variable.capitalize())
    ax.scatter(df[df["multivariate_anomaly"] == 1]["timestamp"],
               df[df["multivariate_anomaly"] == 1][variable],
               color='red', label="Anomaly", zorder=5)
    ax.set_title(f"{variable.capitalize()} with Anomalies")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(variable.capitalize())
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_multivariate(df):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df["timestamp"], df["temperature"], label="Temperature (°C)", color='blue')
    ax1.plot(df["timestamp"], df["precipitation"], label="Precipitation (mm)", color='green')
    ax1.plot(df["timestamp"], df["soil_moisture"], label="Soil Moisture", color='orange')

    ax2 = ax1.twinx()
    ax2.plot(df["timestamp"], df["rainfall_deficit_pct"], label="Rainfall Deficit (%)", color="purple", alpha=0.6)

    anomalies = df[df["multivariate_anomaly"] == 1]
    ax1.scatter(anomalies["timestamp"], anomalies["temperature"], color="red", marker="x", label="Temp Anomaly")
    ax1.scatter(anomalies["timestamp"], anomalies["precipitation"], color="black", marker="o", label="Precip Anomaly")
    ax1.scatter(anomalies["timestamp"], anomalies["soil_moisture"], color="brown", marker="^", label="SoilM Anomaly")

    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Env. Variables")
    ax2.set_ylabel("Rainfall Deficit (%)")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    st.pyplot(fig)

def plot_anomaly_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df[df["multivariate_anomaly"] == 1].timestamp, bins=30, color='red', label="Anomalies")
    ax.hist(df[df["multivariate_anomaly"] == 0].timestamp, bins=30, color='blue', label="Normal")
    ax.set_title("Anomaly Distribution Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)

# -------------------- Main App --------------------
with st.spinner("Fetching and processing data..."):
    data = fetch_live_data()
    if data.empty:
        st.stop()
    check_data_completeness(data)
    rainfall_deficit = calculate_rainfall_deficit(data)
    data, anomaly_ratio = detect_anomalies_multivariate(data, rainfall_deficit, contamination=contamination)

# Sidebar date filter
min_date, max_date = data["timestamp"].min(), data["timestamp"].max()
date_range = st.sidebar.date_input("Date Range", [min_date.date(), max_date.date()])
filtered = data[(data["timestamp"].dt.date >= date_range[0]) & (data["timestamp"].dt.date <= date_range[1])]

# Email alerts for anomalies in the past hour
recent = data[(data["multivariate_anomaly"] == 1) & (data["timestamp"] > datetime.now() - timedelta(hours=1))]
if not recent.empty:
    body = f"Drought anomaly detected at {recent['timestamp'].iloc[-1]}\n\n"
    body += recent[['timestamp', 'temperature', 'precipitation', 'soil_moisture', 'rainfall_deficit_pct']].to_string(index=False)
    send_email_alert("Drought Anomaly Detected", body)

# -------------------- KPIs --------------------
total_records = len(filtered)
total_anomalies = int(filtered["multivariate_anomaly"].sum())
avg_rainfall = filtered["precipitation"].mean()
avg_soil_moisture = filtered["soil_moisture"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 Total Records", total_records)
col2.metric("🚨 Total Anomalies", total_anomalies)
col3.metric("🌧️ Avg Rainfall (mm)", f"{avg_rainfall:.2f}")
col4.metric("🌱 Avg Soil Moisture", f"{avg_soil_moisture:.2f}")

st.sidebar.header("📊 Summary")
st.sidebar.write(f"Multivariate anomalies: {total_anomalies}")
st.sidebar.metric("Anomaly Ratio", f"{anomaly_ratio * 100:.2f}%")

# -------------------- Plots --------------------
st.subheader("📈 Anomaly Visualization")
plot_anomalies(filtered, selected_variable)

st.subheader("📉 Multivariate View")
plot_multivariate(filtered)

st.subheader("📊 Anomaly Distribution")
plot_anomaly_distribution(filtered)

# -------------------- Downloads --------------------
st.subheader("📅 Download Reports")
anomalies = filtered[filtered["multivariate_anomaly"] == 1]
st.download_button("Download Anomaly CSV", anomalies.to_csv(index=False), "anomalies.csv", "text/csv")
st.download_button("Download Full Data CSV", filtered.to_csv(index=False), "all_data.csv", "text/csv")
