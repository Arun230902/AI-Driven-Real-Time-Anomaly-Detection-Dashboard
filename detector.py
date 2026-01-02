import time
import logging
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from prometheus_client import Gauge, start_http_server

# Inside the docker network, Prometheus is reachable via service name
PROM_URL = "http://prometheus:9090"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [detector] %(levelname)s: %(message)s",
)

# Prometheus metrics exposed by this service
ANOMALY_FLAG = Gauge(
    "request_throughput_anomaly_detected",
    "1 if anomaly detected in the last evaluation window, else 0",
)
ACCURACY_G = Gauge(
    "request_throughput_detection_accuracy",
    "Accuracy of anomaly detection over the last evaluation window (0-1)",
)
PRECISION_G = Gauge(
    "request_throughput_detection_precision",
    "Precision of anomaly detection over the last evaluation window (0-1)",
)
RECALL_G = Gauge(
    "request_throughput_detection_recall",
    "Recall of anomaly detection over the last evaluation window (0-1)",
)

def query_range(metric_name: str, minutes: int = 60, step: str = "10s") -> pd.DataFrame:
    """
    Query Prometheus for range vector of `metric_name` over the last `minutes`.
    """
    end = datetime.utcnow()
    start = end - timedelta(minutes=minutes)

    params = {
        "query": metric_name,
        "start": start.timestamp(),
        "end": end.timestamp(),
        "step": step,
    }

    resp = requests.get(f"{PROM_URL}/api/v1/query_range", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data["status"] != "success":
        raise RuntimeError(f"Prometheus query failed: {data}")

    result = data["data"]["result"]
    if not result:
        raise RuntimeError(f"No data returned for metric {metric_name}")

    values = result[0]["values"]

    timestamps = [datetime.fromtimestamp(float(ts)) for ts, _ in values]
    vals = [float(v) for _, v in values]

    df = pd.DataFrame({"timestamp": timestamps, "value": vals})
    return df

def query_two_series(metric_value: str, metric_gt: str, minutes: int = 60, step: str = "10s") -> pd.DataFrame:
    """
    Query value metric and ground-truth metric, align by timestamp.
    """
    df_val = query_range(metric_value, minutes, step)
    df_gt = query_range(metric_gt, minutes, step)

    df_val = df_val.sort_values("timestamp")
    df_gt = df_gt.sort_values("timestamp")

    df = pd.merge_asof(
        df_val,
        df_gt,
        on="timestamp",
        direction="nearest",
        suffixes=("_val", "_gt"),
    )

    df.rename(columns={"value_val": "value", "value_gt": "is_anomaly_gt"}, inplace=True)
    df["is_anomaly_gt"] = df["is_anomaly_gt"].round().astype(int)
    return df

def detect_anomalies_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    mean = df["value"].mean()
    std = df["value"].std() or 1e-9
    df["zscore"] = (df["value"] - mean) / std
    df["is_anomaly_z"] = df["zscore"].abs() > threshold
    return df

def detect_anomalies_isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    model = IsolationForest(contamination=contamination, random_state=42)
    values = df["value"].values.reshape(-1, 1)
    preds = model.fit_predict(values)  # -1 = anomaly
    df["is_anomaly_if"] = preds == -1
    return df

def compute_metrics(df: pd.DataFrame):
    df["is_anomaly"] = df["is_anomaly_z"] | df["is_anomaly_if"]

    df["tp"] = df["is_anomaly"] & (df["is_anomaly_gt"] == 1)
    df["tn"] = (~df["is_anomaly"]) & (df["is_anomaly_gt"] == 0)
    df["fp"] = df["is_anomaly"] & (df["is_anomaly_gt"] == 0)
    df["fn"] = (~df["is_anomaly"]) & (df["is_anomaly_gt"] == 1)

    tp = int(df["tp"].sum())
    tn = int(df["tn"].sum())
    fp = int(df["fp"].sum())
    fn = int(df["fn"].sum())

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return tp, tn, fp, fn, accuracy, precision, recall

def main_loop():
    # Expose this service's metrics on :9100
    start_http_server(9100)
    logging.info("Anomaly-detector Prometheus metrics server started on :9100")

    # initialize metrics
    ANOMALY_FLAG.set(0)
    ACCURACY_G.set(0)
    PRECISION_G.set(0)
    RECALL_G.set(0)

    while True:
        try:
            logging.info("Querying Prometheus for throughput and ground truth...")
            df = query_two_series(
                "request_throughput",
                "request_throughput_is_anomaly_gt",
                minutes=60,
                step="10s",
            )
            logging.info("Fetched %d data points", len(df))

            df = detect_anomalies_zscore(df, threshold=3.0)
            df = detect_anomalies_isolation_forest(df, contamination=0.05)

            tp, tn, fp, fn, accuracy, precision, recall = compute_metrics(df)

            has_anomaly = int(df["is_anomaly"].any())
            ANOMALY_FLAG.set(has_anomaly)
            ACCURACY_G.set(accuracy)
            PRECISION_G.set(precision)
            RECALL_G.set(recall)

            logging.info(
                "TP=%d TN=%d FP=%d FN=%d | Acc=%.2f Prec=%.2f Rec=%.2f | anomaly_flag=%d",
                tp, tn, fp, fn,
                accuracy, precision, recall,
                has_anomaly,
            )

        except Exception as e:
            logging.exception("Error during anomaly detection: %s", e)
            # Do not crash; just report no anomaly this round
            ANOMALY_FLAG.set(0)

        # wait before next evaluation
        time.sleep(60)

if __name__ == "__main__":
    main_loop()
