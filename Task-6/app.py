import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

st.set_page_config(page_title="Stock Forecast with ARIMA", layout="centered")
st.title("Stock Price Forecasting using ARIMA")

uploaded_file = st.file_uploader("Upload your stock data (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # Column selectors
    st.write("Select columns for analysis:")
    date_col = st.selectbox("Date column", df.columns)
    price_col = st.selectbox("Close Price column", df.columns)

    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        data = df[price_col].dropna()

        # Plot the original time series
        st.subheader("📊 Historical Price Plot")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data)
        ax.set_title("Stock Closing Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.grid(True)
        st.pyplot(fig)

        # ADF Test
        result = adfuller(data)
        st.write("**ADF Statistic:**", result[0])
        st.write("**p-value:**", result[1])
        d = 1 if result[1] > 0.05 else 0

        # ARIMA model
        st.write("Fitting ARIMA Model...")
        model = ARIMA(data, order=(5, d, 2))
        model_fit = model.fit()
        st.success("Model fitted successfully!")

        # Forecast
        n_forecast = st.slider("Select days to forecast", 7, 60, 30)
        forecast = model_fit.forecast(steps=n_forecast)
        forecast_dates = pd.date_range(data.index[-1], periods=n_forecast+1, freq='B')[1:]

        st.subheader("📈 Forecast Plot")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(data, label="Historical")
        ax2.plot(forecast_dates, forecast, label="Forecast", color='orange')
        ax2.set_title("ARIMA Forecast")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Close Price")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
