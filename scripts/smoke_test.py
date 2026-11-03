"""
Smoke test for a running API: python scripts/smoke_test.py [base_url]
Waits for /health, then checks /predict. Uses only the standard library, so CI can run it against the container.
"""

import json
import sys
import time
import urllib.request

CUSTOMER = {
    "months": 11, "eqpdays": 700.0, "change_mou": -400.0, "mou_Mean": 80.0, "totmrc_Mean": 30.0,
    "avgqty": 60.0, "hnd_price": 9.99, "uniqsubs": 1, "drop_vce_Mean": 1.0, "change_rev": -20.0,
    "mou_cvce_Mean": 30.0, "avgrev": 30.0, "ovrmou_Mean": 0.0, "actvsubs": 1, "asl_flag": "N",
}


def wait_for_health(base_url: str, attempts: int = 60) -> dict:
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as res:
                return json.load(res)
        except OSError:  # not up yet, or 503 while degraded
            time.sleep(2)
    sys.exit("API did not become healthy in time")


def main() -> None:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    health = wait_for_health(base_url)
    print("health:", health)

    request = urllib.request.Request(
        f"{base_url}/predict",
        data=json.dumps(CUSTOMER).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as res:
        prediction = json.load(res)
    print("prediction:", prediction["probability"], prediction["risk_level"])

    if prediction["risk_level"] != "high":
        sys.exit(f"Expected a high-risk result, got {prediction['risk_level']}")


if __name__ == "__main__":
    main()
