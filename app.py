
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Electricity Forecasting",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Household Electricity Consumption Forecasting")
st.write(
    "LSTM deep learning model for predicting "
    "the next minute of electricity consumption."
)

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        "household_lstm_model.keras"
    )
    feature_scaler = joblib.load("feature_scaler.pkl")
    target_scaler = joblib.load("target_scaler.pkl")
    return model, feature_scaler, target_scaler

try:
    model, feature_scaler, target_scaler = load_artifacts()
    st.success("LSTM model loaded successfully!")
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

feature_cols = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

st.subheader("Upload electricity measurements")

st.write(
    "Upload a CSV containing at least 60 consecutive "
    "minute-level records and all seven measurement columns."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        missing = [
            col for col in feature_cols
            if col not in data.columns
        ]

        if missing:
            st.error(
                "Missing required columns: "
                + ", ".join(missing)
            )
            st.stop()

        data[feature_cols] = data[feature_cols].apply(
            pd.to_numeric, errors="coerce"
        )

        data[feature_cols] = data[feature_cols].interpolate(
            method="linear",
            limit_direction="both"
        )

        data = data.dropna(subset=feature_cols)

        if len(data) < 60:
            st.error(
                "Please upload at least 60 valid records."
            )
            st.stop()

        st.write("Uploaded data preview")
        st.dataframe(data.head())

        if st.button("Predict next-minute consumption"):
            # Use the most recent 60 records
            recent_data = data[feature_cols].tail(60).values

            scaled_data = feature_scaler.transform(
                recent_data
            )

            X = scaled_data.reshape(1, 60, 7)

            prediction_scaled = model.predict(
                X, verbose=0
            )

            prediction = target_scaler.inverse_transform(
                prediction_scaled
            )[0, 0]

            st.subheader("Prediction result")

            st.metric(
                "Predicted Global Active Power",
                f"{prediction:.3f} kW"
            )

            st.caption(
                "This is a one-minute-ahead estimate "
                "based on the last 60 uploaded records."
            )

            fig, ax = plt.subplots(figsize=(10, 4))

            ax.plot(
                range(len(data)),
                data["Global_active_power"],
                label="Historical consumption"
            )

            ax.scatter(
                len(data),
                prediction,
                color="red",
                label="Next-minute prediction"
            )

            ax.set_title("Historical Data and Prediction")
            ax.set_xlabel("Record")
            ax.set_ylabel("Global Active Power (kW)")
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing the CSV: {e}")

st.divider()
st.caption(
    "Academic project | Deep Learning | LSTM "
    "| Household Electricity Forecasting"
)
