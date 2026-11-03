"""
Smoke test for a running API: python scripts/smoke_test.py [base_url]
Waits for /health, then checks /predict. Uses only the standard library, so CI can run it against the container.
"""

import json
import sys
import time
import urllib.request

CUSTOMER = {
    "gender": "Female", "senior_citizen": 0, "partner": "No", "dependents": "No",
    "tenure": 1, "contract": "Month-to-month", "paperless_billing": "Yes",
    "payment_method": "Electronic check", "monthly_charges": 85.0,
    "phone_service": "Yes", "multiple_lines": "No", "internet_service": "Fiber optic",
    "online_security": "No", "online_backup": "No", "device_protection": "No",
    "tech_support": "No", "streaming_tv": "No", "streaming_movies": "No",
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
