Perfect 👍 — since you’ve used **scikit-learn** (for the trendline regression model inside `plotly.express` and potentially future ML extensions), here’s the **updated `README.md`** including it properly in the **Tech Stack**, **Requirements**, and **Enhancements** sections 👇

---

# 💰 Stock Market Insights Dashboard

### 📊 Real-Time Stock Analysis using Streamlit, Plotly, yFinance & scikit-learn

This project is an **interactive dashboard** built using **Streamlit** that provides **real-time insights** into the performance, volatility, and behavior of major Indian stocks (e.g., TCS, Infosys, HDFC Bank, Reliance, ITC).
It helps users visualize **price trends, trading volume, volatility patterns, and correlations** through a clean, interactive interface.

---

## 🚀 Features

✅ **Live Data Fetching** from Yahoo Finance using `yfinance`

✅ **Candlestick Chart** with Moving Averages (MA20 & MA50)

✅ **Volume Trend Analysis**

✅ **Daily Returns Distribution**

✅ **Rolling Volatility (20-day)**

✅ **Volume vs Closing Price Correlation** (using Linear Regression with scikit-learn via Plotly)

✅ **Correlation Heatmap** for Market Indicators

✅ **Dynamic Insights Section** with Key Metrics

---

## 🧠 Tech Stack

| Component            | Technology Used                                |
| -------------------- | ---------------------------------------------- |
| Dashboard Framework  | Streamlit                                      |
| Data Source          | Yahoo Finance (`yfinance`)                     |
| Visualization        | Plotly (Graph Objects & Express)               |
| Data Handling        | Pandas, NumPy                                  |
| Statistical Modeling | scikit-learn (Linear Regression for trendline) |
| Analytical Tools     | Correlation Heatmap & Rolling Volatility       |

---

## 🧩 Project Structure

```
📁 Stock-Market-Dashboard/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── screenshots/          # (Optional) Dashboard snapshots
```

---

## ⚙️ Installation & Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/<your-username>/Stock-Market-Dashboard.git
   cd Stock-Market-Dashboard
   ```

2. **Create and Activate Virtual Environment**

   ```bash
   python -m venv .env
   .\.env\Scripts\activate      # For Windows
   source .env/bin/activate     # For Mac/Linux
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**

   ```bash
   streamlit run app.py
   ```

---

## 🧮 Example Stocks

You can analyze the following tickers:

* `TCS.NS` – Tata Consultancy Services
* `INFY.NS` – Infosys
* `HDFCBANK.NS` – HDFC Bank
* `RELIANCE.NS` – Reliance Industries
* `ITC.NS` – ITC Limited

---

## 📸 Dashboard Overview

| Visualization               | Description                                                           |
| --------------------------- | --------------------------------------------------------------------- |
| **Candlestick + MA Lines**  | Shows stock price movements with short & long-term trends             |
| **Volume Bars**             | Indicates trading activity over time                                  |
| **Returns Histogram**       | Displays frequency of daily returns                                   |
| **Volatility Line Chart**   | Highlights market uncertainty over time                               |
| **Volume vs Price Scatter** | Shows how price reacts to trading volume (trendline via scikit-learn) |
| **Correlation Heatmap**     | Reveals interdependence between indicators                            |

---

## 🧠 Insights Generated

* Identifies **bullish/bearish trends** using moving averages.
* Highlights **high volatility periods**.
* Analyzes how **volume impacts price fluctuations**.
* Evaluates **correlation** between price, volume, and volatility metrics.

---

## 📦 Requirements

```
streamlit
pandas
numpy
plotly
yfinance
scikit-learn
```

Create this file as `requirements.txt`.

---

## 📈 Future Enhancements

* Integrate **more predictive models** (LSTM, Prophet, ARIMA) for trend forecasting
* Add **sentiment analysis** from financial news
* Include **technical indicators** (RSI, MACD, Bollinger Bands)
* Enable **PDF/CSV report export**
* Support **multi-stock comparison dashboards**

---

## 👨‍💻 Author

**Sanjai M**
B.E. Computer Science and Engineering (AI & ML)
KPR Institute of Engineering and Technology

📬 *If you like this project, give it a ⭐ on GitHub!*

---
