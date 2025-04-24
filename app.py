import datetime
import os

import pandas as pd
import plotly.graph_objs as go
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/LSTM_Predict"

MIN_DATE = datetime.date(2020, 1, 1)
MAX_DATE = datetime.date(2022, 12, 31)

DATA_DIR = "data"  # assume all CSVs are here


def main():
    st.title("Stock Price Predictor")

    stock_name = st.selectbox(
        "Please choose stock name", ("AAPL", "TSLA", "AMZN", "MSFT")
    )

    start_date = st.date_input(
        "Start date", min_value=MIN_DATE, max_value=MAX_DATE, value=MIN_DATE
    )
    end_date = st.date_input(
        "End date", min_value=MIN_DATE, max_value=MAX_DATE, value=MAX_DATE
    )

    if start_date > end_date:
        st.error("End date must be after start date.")
        return

    # Load local data file
    csv_path = os.path.join(DATA_DIR, f"{stock_name}_data.csv")
    if not os.path.exists(csv_path):
        st.error(f"No local data found for {stock_name} at {csv_path}")
        return

    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df[(df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))]

    if df.empty:
        st.warning("No data available in the selected date range.")
        return

    # Plot actual stock prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Actual Close Price"))
    fig.update_layout(title=f"{stock_name} Stock Price", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

    if st.button("Predict"):
        payload = {"stock_name": stock_name}
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()

            predictions = response.json()["prediction"]

            # Ensure we only take as many dates as predictions
            pred_dates = df["Date"].iloc[-len(predictions):]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Actual"))
            fig.add_trace(go.Scatter(x=pred_dates, y=predictions, name="Predicted"))
            fig.update_layout(
                title=f"{stock_name} - Actual vs Predicted",
                xaxis_title="Date",
                yaxis_title="Price"
            )
            st.plotly_chart(fig)

        except requests.exceptions.RequestException as e:
            st.error(f"Error occurred while making the prediction request:\n\n{e}")


if __name__ == "__main__":
    main()
