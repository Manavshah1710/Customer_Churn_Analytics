#!/usr/bin/env python
"""
Benchmark script for SHAP explainability.
Measures: import shap, explainer build, cold explain_customer, warm explain_customer.
"""

import time
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.explain import get_explainer, explain_customer


def main():
    # Test import time
    import_start = time.perf_counter()
    import shap
    import_time = time.perf_counter() - import_start
    print(f"import shap: {import_time*1000:.1f}ms")

    # Test explainer build
    build_start = time.perf_counter()
    explainer = get_explainer()
    build_time = time.perf_counter() - build_start
    print(f"explainer build: {build_time*1000:.1f}ms")

    # Test customers
    customer_a = {
        "gender": "Female", "senior_citizen": 0, "partner": "No", "dependents": "No",
        "tenure": 1, "contract": "Month-to-month", "paperless_billing": "Yes",
        "payment_method": "Electronic check", "monthly_charges": 85.0, "total_charges": 85.0,
        "phone_service": "Yes", "multiple_lines": "No", "internet_service": "Fiber optic",
        "online_security": "No", "online_backup": "No", "device_protection": "No",
        "tech_support": "No", "streaming_tv": "No", "streaming_movies": "No",
    }

    customer_b = {
        "gender": "Male", "senior_citizen": 0, "partner": "Yes", "dependents": "Yes",
        "tenure": 60, "contract": "Two year", "paperless_billing": "No",
        "payment_method": "Credit card (automatic)", "monthly_charges": 60.0, "total_charges": 3600.0,
        "phone_service": "Yes", "multiple_lines": "Yes", "internet_service": "DSL",
        "online_security": "Yes", "online_backup": "Yes", "device_protection": "Yes",
        "tech_support": "Yes", "streaming_tv": "No", "streaming_movies": "No",
    }

    # Cold call (first call after build)
    cold_start = time.perf_counter()
    exp_a = explain_customer(customer_a)
    cold_time = time.perf_counter() - cold_start
    print(f"cold explain_customer: {cold_time*1000:.1f}ms")

    # Warm call
    warm_start = time.perf_counter()
    exp_b = explain_customer(customer_b)
    warm_time = time.perf_counter() - warm_start
    print(f"warm explain_customer: {warm_time*1000:.1f}ms")

    print(f"\nSummary:")
    print(f"  import shap:        {import_time*1000:>8.1f}ms")
    print(f"  explainer build:    {build_time*1000:>8.1f}ms")
    print(f"  cold explain:       {cold_time*1000:>8.1f}ms")
    print(f"  warm explain:       {warm_time*1000:>8.1f}ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())