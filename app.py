import time
import random
from prometheus_client import Gauge, start_http_server

# numeric throughput
THROUGHPUT = Gauge('request_throughput', 'Simulated requests per second')
# ground truth label: 1 if this point is an injected anomaly, else 0
IS_ANOMALY_GT = Gauge(
    'request_throughput_is_anomaly_gt',
    'Ground truth: 1 if this point is an injected anomaly spike, else 0'
)

def generate_traffic():
    base = 100  # normal around 100 req/s
    while True:
        value = base + random.randint(-10, 10)  # normal noise
        is_anomaly = 0

        # 5% chance to inject a big spike anomaly
        if random.random() < 0.5:
            value *= random.randint(3, 6)
            is_anomaly = 1
            print(f"[metrics-app] Injected anomaly spike: {value} req/s")
        else:
            print(f"[metrics-app] Normal value: {value} req/s")

        THROUGHPUT.set(value)
        IS_ANOMALY_GT.set(is_anomaly)

        time.sleep(5)  # matches Prometheus scrape_interval

if __name__ == "__main__":
    start_http_server(8000)
    print("[metrics-app] Metrics HTTP server started on :8000")
    generate_traffic()
