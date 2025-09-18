# 🌾 Real-Time Drought Anomaly Detection

A **Streamlit-based application** that detects drought anomalies in real time using **Open-Meteo API** (weather + soil data), **machine learning anomaly detection**, and provides **visual insights & email alerts**. Designed for early drought warning systems, ideal for hackathons and real-world deployment.

---

## 🚀 Features

* ✅ Real-time weather & soil moisture data from Open-Meteo API
* ✅ Machine learning anomaly detection (Isolation Forest)
* ✅ Rolling averages & rainfall deficit calculations
* ✅ Interactive Streamlit dashboard with charts
* ✅ Email notifications for anomaly alerts
* ✅ Docker-ready for deployment anywhere

---

## 🛠 Tech Stack

* **Frontend/UI**: Streamlit
* **Backend**: Python
* **ML Model**: Scikit-learn (Isolation Forest)
* **API**: Open-Meteo API
* **Deployment**: Docker / Streamlit Cloud

---

## 📂 Project Structure

```
.
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── Dockerfile          # Containerization
├── README.md           # Documentation
└── .env.example        # Environment variables template (for email alerts)
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repo

```bash
git clone https://github.com/your-username/realtime-drought-detection.git
cd realtime-drought-detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Locally

```bash
streamlit run app.py
```

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t drought-detection .
```

### Run Container

```bash
docker run -p 8501:8501 drought-detection
```

Visit 👉 [http://localhost:8501](http://localhost:8501)

---

## ☁️ Streamlit Cloud Deployment

1. Push your repo to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. Deploy by selecting your repo + `app.py`

---

## 🔑 Environment Variables

Create a `.env` file for email alerts:

```
EMAIL_USER=your_email@gmail.com
EMAIL_PASS=your_password
TO_EMAIL=recipient_email@gmail.com
```

*(Never commit real credentials – use `.env.example`)*

---


## 📖 References

* [Open-Meteo API Docs](https://open-meteo.com/)
* [Streamlit Documentation](https://docs.streamlit.io/)
* [Scikit-learn Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)

---

👩‍💻 Developed by **Neha Shree** 
