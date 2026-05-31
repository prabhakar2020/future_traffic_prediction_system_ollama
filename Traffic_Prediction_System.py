"""
AI Medical Assistant Without Severity Triage
# I am running this script on my Ubutnu
# Python 3.12.3
# Description:    Ubuntu 24.04.4 LTS
# Release:        24.04
# Codename:       noble

"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Data Generation (Some realistic city traffic pattern)
def generate_data(days=360):
    print("Generating historical traffic dataset for 365 days")

    date_range = pd.date_range(
        start="2025-01-01",
        periods=days * 24,
        freq="h"
    )

    df = pd.DataFrame()
    df["datetime"] = date_range

    # Time based data
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    df["is_peak_hour"] = df["hour"].apply(lambda x: 1 if x in [8, 9, 10, 17, 18, 19] else 0)

    # Human activity based data
    df["school_open"] = np.where(df["is_weekend"] == 1, 0, 1)
    df["college_open"] = 1

    # events based data
    df["festival"] = np.where(df.index % 700 == 0, 1, 0)
    df["political_event"] = np.where(df.index % 1000 == 0, 1, 0)
    df["mall_opening"] = np.where(df.index % 1200 == 0, 1, 0)
    df["special_event"] = np.where(df.index % 500 == 0, 1, 0)

    # weather based data
    df["rain"] = np.random.choice([0, 1, 2], size=len(df), p=[0.7, 0.2, 0.1])

    # Realistic traffic model

    base = 30
    morning_peak = np.where(df["hour"].between(8, 10), 25, 0)
    evening_peak = np.where(df["hour"].between(17, 20), 60, 0)
    lunch_effect = np.where(df["hour"].between(12, 14), 10, 0)
    night_drop = np.where(df["hour"].between(21, 23), -15, 0)
    early_morning = np.where(df["hour"].between(0, 6), -20, 0)

    weekday_effect = np.where(df["is_weekend"] == 0, 15, -5)

    human_effect = (
        df["school_open"] * np.where(df["hour"].between(7, 9), 20, 5) +
        df["college_open"] * 10
    )

    event_effect = (
        df["festival"] * 120 +
        df["political_event"] * 100 +
        df["mall_opening"] * 80 +
        df["special_event"] * 130
    )

    weather_effect = (
        np.where(df["rain"] == 1, 25, 0) +
        np.where(df["rain"] == 2, 60, 0)
    )

    random_spike = np.random.choice(
        [0, 0, 5, 10, 20, 40, 90],
        size=len(df),
        p=[0.4, 0.2, 0.15, 0.1, 0.08, 0.05, 0.02]
    )

    traffic = (
        base +
        morning_peak +
        evening_peak +
        lunch_effect +
        night_drop +
        early_morning +
        weekday_effect +
        human_effect +
        event_effect +
        weather_effect +
        random_spike +
        np.random.randint(-3, 4, size=len(df))
    )

    df["traffic_index"] = traffic

    df = df.drop(columns=["datetime"])
    # Historical traffic data stored in csv file
    file_name = "historical_traffic_data.csv"
    df.to_csv(file_name, index=False)
    print(f"Dataset saved as: {file_name}")

    print("Dataset generation completed.")
    return df


# Train the model using RandomForest 
def train_model(df):
    X = df.drop(columns=["traffic_index"])
    y = df["traffic_index"]
    feature_columns = X.columns.tolist()
    x_train_val, x_temp_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(x_train_val, y_train_val)
    print("Model training completed.")
    return model, feature_columns


# predict hourly traffic for future date
def predict_tomorrow_hourly(model, feature_columns):
    tomorrow = datetime.now() + timedelta(days=1)
    date_str = tomorrow.strftime("%Y-%d-%m")
    results = []
    print("\nHOURLY TRAFFIC FORECAST (8 AM - 11 PM)\n")
    for hour in range(8, 24):

        sample = pd.DataFrame([{
            "hour": hour,
            "day_of_week": tomorrow.weekday(),
            "is_weekend": 1 if tomorrow.weekday() >= 5 else 0,
            "is_peak_hour": 1 if hour in [8, 9, 10, 17, 18, 19] else 0,
            "school_open": 1,
            "college_open": 1,
            "festival": 0,
            "political_event": 0,
            "mall_opening": 0,
            "special_event": 0,
            "rain": 1
        }])

        sample = sample.reindex(columns=feature_columns, fill_value=0)

        prediction = model.predict(sample)[0]

        # generate final traffic status
        if prediction >= 160:
            status = "VERY HIGH TRAFFIC"
        elif prediction >= 110:
            status = "HIGH TRAFFIC"
        elif prediction >= 70:
            status = "MODERATE TRAFFIC"
        else:
            status = "LOW TRAFFIC"

        results.append([date_str, hour, prediction, status])

        print(f"{date_str} {hour}:00 → {prediction:.2f} → {status}")

    return pd.DataFrame(results, columns=["Date", "Hour", "Traffic_Index", "Status"])

if __name__ == "__main__":
    df = generate_data()
    model, feature_columns = train_model(df)
    forecast = predict_tomorrow_hourly(model, feature_columns)
    print("\n================ FINAL OUTPUT ================\n")
    print(forecast)
