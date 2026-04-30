import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def preprocess_flows(df: pd.DataFrame):
    df = df.copy()

    df = df.drop_duplicates()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.fillna(df.median(numeric_only=True), inplace=True)

    if "protocol" in df.columns:
        le = LabelEncoder()
        df["protocol_encoded"] = le.fit_transform(df["protocol"].astype(str))
    else:
        df["protocol_encoded"] = 0

    if "label" not in df.columns:
        df["label"] = 0

    if "timestamp" not in df.columns:
        df["timestamp"] = range(len(df))

    required_numeric = ["flow_duration", "total_packets", "total_bytes", "protocol_encoded"]
    for col in required_numeric:
        if col not in df.columns:
            df[col] = 0

    scaler = MinMaxScaler()
    df[required_numeric] = scaler.fit_transform(df[required_numeric])

    return df, scaler