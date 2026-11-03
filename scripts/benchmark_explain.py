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
from src.schema import EXAMPLE_HIGH_RISK, EXAMPLE_LOW_RISK


def main():
    # Test import time
    import_start = time.perf_counter()
    import shap  # noqa: F401 - imported only to time it
    import_time = time.perf_counter() - import_start
    print(f"import shap: {import_time*1000:.1f}ms")

    # Test explainer build
    build_start = time.perf_counter()
    get_explainer()
    build_time = time.perf_counter() - build_start
    print(f"explainer build: {build_time*1000:.1f}ms")

    # Test customers
    customer_a = EXAMPLE_HIGH_RISK
    customer_b = EXAMPLE_LOW_RISK

    # Cold call (first call after build)
    cold_start = time.perf_counter()
    explain_customer(customer_a)
    cold_time = time.perf_counter() - cold_start
    print(f"cold explain_customer: {cold_time*1000:.1f}ms")

    # Warm call
    warm_start = time.perf_counter()
    explain_customer(customer_b)
    warm_time = time.perf_counter() - warm_start
    print(f"warm explain_customer: {warm_time*1000:.1f}ms")

    print("\nSummary:")
    print(f"  import shap:        {import_time*1000:>8.1f}ms")
    print(f"  explainer build:    {build_time*1000:>8.1f}ms")
    print(f"  cold explain:       {cold_time*1000:>8.1f}ms")
    print(f"  warm explain:       {warm_time*1000:>8.1f}ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())