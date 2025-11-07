import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------
# Basic Config
# ---------------------------------------------------
st.set_page_config(page_title="Stock Market Insights Dashboard", layout="wide")
st.title("💰 Stock Market Insights Dashboard")
st.markdown("Analyze stock performance, volatility, and trends interactively.")

# ---------------------------------------------------
# Sidebar filters
# ---------------------------------------------------
stocks = ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "RELIANCE.NS", "ITC.NS"]
selected_stock = st.sidebar.selectbox("📈 Select Stock", stocks)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

st.info(f"Fetching live data for *{selected_stock}*...")
raw = yf.download(selected_stock, start=start_date, end=end_date)

# Flatten MultiIndex columns if needed
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = [col[0] for col in raw.columns]

raw.reset_index(inplace=True)
data = raw.copy()

# ---------------------------------------------------
# Compute Indicators
# ---------------------------------------------------
data["MA20"] = data["Close"].rolling(window=20).mean()
data["MA50"] = data["Close"].rolling(window=50).mean()
data["Daily Return"] = data["Close"].pct_change() * 100
data["Volatility(20d)"] = data["Daily Return"].rolling(window=20).std()

# ---------------------------------------------------
# Subplot 1: Price + Volume
# ---------------------------------------------------
st.subheader("Price & Volume Overview")

fig1 = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.03, row_heights=[0.7, 0.3],
    subplot_titles=("Price Movement with MA20 & MA50", "Trading Volume")
)

fig1.add_trace(go.Candlestick(
    x=data["Date"], open=data["Open"], high=data["High"],
    low=data["Low"], close=data["Close"], name="Candlestick"
), row=1, col=1)

fig1.add_trace(go.Scatter(
    x=data["Date"], y=data["MA20"], line=dict(color="orange", width=1.5), name="MA 20"
), row=1, col=1)

fig1.add_trace(go.Scatter(
    x=data["Date"], y=data["MA50"], line=dict(color="blue", width=1.5), name="MA 50"
), row=1, col=1)

fig1.add_trace(go.Bar(
    x=data["Date"], y=data["Volume"], marker_color="rgba(0,0,150,0.4)", name="Volume"
), row=2, col=1)

fig1.update_layout(height=600, showlegend=True, xaxis_rangeslider_visible=False, template="plotly_white")
st.plotly_chart(fig1, use_container_width=True)

# ---------------------------------------------------
# Subplot 2: Returns & Volatility
# ---------------------------------------------------
st.subheader("Returns & Volatility Analysis")
col1, col2 = st.columns(2)

with col1:
    fig2 = px.histogram(data, x="Daily Return", nbins=50, color_discrete_sequence=["#FF6347"],
                        title="Distribution of Daily Returns (%)")
    fig2.update_layout(template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    fig3 = px.line(data, x="Date", y="Volatility(20d)",
                   title="20-Day Rolling Volatility (%)",
                   color_discrete_sequence=["#4682B4"])
    fig3.update_layout(template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------
# FIXED: Volume vs Closing Price Correlation
# ---------------------------------------------------
st.subheader("Volume vs Closing Price Correlation")

df_vp = data[["Volume", "Close"]].dropna()
df_vp = df_vp[df_vp["Volume"] > 0]

x = np.log10(df_vp["Volume"])
y = df_vp["Close"]

slope, intercept = np.polyfit(x, y, 1)
r2 = r2_score(y, slope * x + intercept)

fig4 = px.scatter(
    x=x, y=y, trendline="ols", title=f"Volume vs Closing Price (Log Scale) — R² = {r2:.3f}",
    labels={"x": "Log10(Volume)", "y": "Closing Price (₹)"}
)
fig4.add_trace(go.Scatter(x=x, y=slope*x + intercept, mode="lines", name="Linear Fit", line=dict(color="red")))
fig4.update_layout(template="plotly_white")
st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------
st.subheader("Feature Correlation Heatmap")

corr = data[["Open", "High", "Low", "Close", "Volume", "Daily Return", "Volatility(20d)"]].corr()
fig5 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                 title="Correlation Between Market Indicators")
fig5.update_layout(template="plotly_white", height=500)
st.plotly_chart(fig5, use_container_width=True)

# ---------------------------------------------------
# Quick Insights
# ---------------------------------------------------
st.subheader("Quick Insights")
latest = data.iloc[-1]
prev = data.iloc[-2]
change = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100
vol_trend = "increased" if latest["Volume"] > data["Volume"].mean() else "decreased"

st.markdown(f"""
- *Last Closing Price:* ₹{latest['Close']:.2f}  
- *Daily Change:* {change:.2f}%  
- *Avg 20-day Volatility:* {data['Volatility(20d)'].mean():.2f}%  
- *Trading Volume has {vol_trend} compared to 30-day average.*
""")

# ---------------------------------------------------
# ML Models Section
# ---------------------------------------------------
st.header("Predictive Modeling")

model_option = st.selectbox(
    "Select Model:", 
    ["Linear Regression", "Random Forest Regressor", "ARIMA Time-Series Forecast"]
)

data_model = data.dropna().copy()
data_model["MA10"] = data_model["Close"].rolling(10).mean()
data_model["MA30"] = data_model["Close"].rolling(30).mean()
data_model = data_model.dropna()

if model_option == "Linear Regression":
    X = data_model[["Open", "High", "Low", "Volume", "MA10", "MA30"]]
    y = data_model["Close"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"*MSE:* {mse:.2f} | *R²:* {r2:.3f}")

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(y=y_test, mode="lines", name="Actual"))
    fig6.add_trace(go.Scatter(y=y_pred, mode="lines", name="Predicted"))
    fig6.update_layout(title="Linear Regression: Actual vs Predicted", template="plotly_white")
    st.plotly_chart(fig6, use_container_width=True)

elif model_option == "Random Forest Regressor":
    X = data_model[["Open", "High", "Low", "Volume", "MA10", "MA30"]]
    y = data_model["Close"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"*MSE:* {mse:.2f} | *R²:* {r2:.3f}")

    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(y=y_test, mode="lines", name="Actual"))
    fig7.add_trace(go.Scatter(y=y_pred, mode="lines", name="Predicted", line=dict(color="green")))
    fig7.update_layout(title="Random Forest: Actual vs Predicted", template="plotly_white")
    st.plotly_chart(fig7, use_container_width=True)

    imp = pd.Series(model.feature_importances_, index=X.columns)
    fig_imp = px.bar(imp, x=imp.index, y=imp.values, title="Feature Importance", labels={"x":"Feature","y":"Importance"})
    fig_imp.update_layout(template="plotly_white")
    st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.subheader("ARIMA Forecasting Model")
    ts = data_model.set_index("Date")["Close"]
    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]
    try:
        model = ARIMA(train, order=(5, 1, 0))
        fit = model.fit()
        forecast = fit.forecast(steps=len(test))

        mse = mean_squared_error(test, forecast)
        st.write(f"*MSE:* {mse:.2f}")

        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(x=train.index, y=train, name="Train"))
        fig8.add_trace(go.Scatter(x=test.index, y=test, name="Actual"))
        fig8.add_trace(go.Scatter(x=test.index, y=forecast, name="Forecast", line=dict(color="red")))
        fig8.update_layout(title="ARIMA Forecasting", template="plotly_white")
        st.plotly_chart(fig8, use_container_width=True)
    except Exception as e:
        st.error(f"ARIMA failed: {e}")