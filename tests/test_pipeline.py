"""
Unit and integration tests for Customer Churn Analytics preprocessing and modeling pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocess import build_features, load_data, split_data, transform_new


@pytest.fixture(scope="module")
def raw_dataset():
    """Loads raw dataset from sqlite database."""
    return load_data("data/churn.db")


@pytest.fixture(scope="module")
def split_dataset(raw_dataset):
    """Splits dataset into train/test."""
    return split_data(raw_dataset, target_col="churn", test_size=0.2, random_state=42)


@pytest.fixture(scope="module")
def processed_dataset(split_dataset):
    """Processes features and fits ColumnTransformer on train only."""
    X_train, X_test, y_train, y_test = split_dataset
    X_train_proc, X_test_proc, artifacts = build_features(X_train, X_test)
    return X_train_proc, X_test_proc, artifacts


def test_no_train_test_overlap(split_dataset):
    """Asserts that there is zero index overlap between training and testing splits."""
    X_train, X_test, y_train, y_test = split_dataset
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)

    assert len(train_indices.intersection(test_indices)) == 0, "Train and test splits share overlapping indices!"
    assert len(X_train) + len(X_test) == len(train_indices) + len(test_indices)


def test_scaler_fit_on_train_only(raw_dataset, split_dataset, processed_dataset):
    """Verifies that the StandardScaler was fit strictly on the training set and not on full/test data."""
    X_train, X_test, y_train, y_test = split_dataset
    X_train_proc, X_test_proc, artifacts = processed_dataset

    # Extract scaler and imputer from ColumnTransformer pipeline
    preprocessor = artifacts["preprocessor"]
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    num_cols = artifacts["num_cols"]

    # Calculate expected means and standard deviations from X_train (median-imputed)
    for i, col in enumerate(num_cols):
        median_val = imputer.statistics_[i]
        train_series = X_train[col].fillna(median_val)
        expected_mean = train_series.mean()
        expected_std = train_series.std(ddof=0)

        # Compare with scaler fitted values
        np.testing.assert_allclose(
            scaler.mean_[i], expected_mean, rtol=1e-5,
            err_msg=f"Scaler mean for {col} does not match X_train mean."
        )
        np.testing.assert_allclose(
            scaler.scale_[i], expected_std, rtol=1e-5,
            err_msg=f"Scaler scale for {col} does not match X_train std."
        )

    # Verify these differ from full-dataset values (leakage check)
    for i, col in enumerate(num_cols):
        full_series = raw_dataset[col].fillna(raw_dataset[col].median())
        full_mean = full_series.mean()
        full_std = full_series.std(ddof=0)
        assert not np.isclose(scaler.mean_[i], full_mean, atol=1e-4), (
            f"Scaler mean for {col} unexpectedly equals full dataset mean."
        )
        assert not np.isclose(scaler.scale_[i], full_std, atol=1e-4), (
            f"Scaler scale for {col} unexpectedly equals full dataset std."
        )


def test_test_class_ratio(raw_dataset, split_dataset):
    """Verifies that the stratified test split preserves the original class ratio within 1%."""
    X_train, X_test, y_train, y_test = split_dataset

    full_churn_ratio = raw_dataset["churn"].mean()
    test_churn_ratio = y_test.mean()
    train_churn_ratio = y_train.mean()

    assert abs(full_churn_ratio - test_churn_ratio) < 0.01, (
        f"Test class ratio ({test_churn_ratio:.4f}) deviates from full data ratio ({full_churn_ratio:.4f})"
    )
    assert abs(train_churn_ratio - test_churn_ratio) < 0.01, (
        f"Train class ratio ({train_churn_ratio:.4f}) deviates from test data ratio ({test_churn_ratio:.4f})"
    )


def test_imputer_statistics_match_train_medians(raw_dataset, split_dataset, processed_dataset):
    """Verifies that the imputer statistics_ equal the medians of X_train (not full dataset)."""
    X_train, X_test, y_train, y_test = split_dataset
    _, _, artifacts = processed_dataset

    preprocessor = artifacts["preprocessor"]
    imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    num_cols = artifacts["num_cols"]

    # Assert imputer statistics_ equal the medians of X_train
    for i, col in enumerate(num_cols):
        expected_train_median = float(X_train[col].median())
        np.testing.assert_allclose(
            imputer.statistics_[i],
            expected_train_median,
            rtol=1e-5,
            err_msg=f"imputer.statistics_ for {col} does not match X_train median",
        )
        np.testing.assert_allclose(
            artifacts["train_medians"][col],
            expected_train_median,
            rtol=1e-5,
            err_msg=f"train_medians[{col}] does not match X_train median",
        )

    # Sanity check: confirm train medians are computed from X_train
    for i, col in enumerate(num_cols):
        assert imputer.statistics_[i] == float(X_train[col].median())


def test_transform_new_feature_columns_and_order(split_dataset, processed_dataset):
    """Verifies transform_new returns exact feature columns in identical order for single-row and batch input."""
    X_train, X_test, y_train, y_test = split_dataset
    _, _, artifacts = processed_dataset

    feature_cols = artifacts["feature_columns"]

    # 1. Single row input
    single_row = X_test.iloc[[0]]
    transformed_single = transform_new(single_row, artifacts)

    assert isinstance(transformed_single, pd.DataFrame), "transform_new must return a DataFrame"
    assert list(transformed_single.columns) == feature_cols, "Single-row column names or order mismatch."
    assert len(transformed_single) == 1, "Single-row output row count mismatch."
    assert transformed_single.isna().sum().sum() == 0, "Single-row transformed output contains NaNs."

    # 2. Batch input
    batch_rows = X_test.head(10)
    transformed_batch = transform_new(batch_rows, artifacts)

    assert isinstance(transformed_batch, pd.DataFrame), "transform_new must return a DataFrame for batch input"
    assert list(transformed_batch.columns) == feature_cols, "Batch column names or order mismatch."
    assert len(transformed_batch) == 10, "Batch output row count mismatch."
    assert transformed_batch.isna().sum().sum() == 0, "Batch transformed output contains NaNs."

    # 3. ColumnTransformer get_feature_names_out matches saved feature_columns
    preprocessor = artifacts["preprocessor"]
    assert preprocessor.get_feature_names_out().tolist() == feature_cols


def test_transform_new_parity_with_batch_test(split_dataset, processed_dataset):
    """
    Parity test: Takes raw rows from test split, transforms each individually with transform_new,
    and asserts the result exactly equals the corresponding batch-processed row.
    """
    X_train, X_test, y_train, y_test = split_dataset
    _, X_test_proc, artifacts = processed_dataset  # X_test_proc is DataFrame from build_features

    num_samples_to_check = min(100, len(X_test))
    for i in range(num_samples_to_check):
        raw_row = X_test.iloc[[i]]
        expected_proc_row = X_test_proc.iloc[[i]]
        single_proc_row = transform_new(raw_row, artifacts)

        pd.testing.assert_frame_equal(
            single_proc_row,
            expected_proc_row,
            check_dtype=False,
            check_exact=False,
            rtol=1e-5,
            atol=1e-5,
        )


def test_transform_new_with_unseen_and_rare_categories(processed_dataset):
    """
    Verifies that a single-row input with unseen or rare categorical values
    still transforms gracefully without error and returns exact feature columns.
    """
    _, _, artifacts = processed_dataset

    unseen_row = pd.DataFrame([{
        "gender": "NonBinary",
        "senior_citizen": 1,
        "partner": "Unknown",
        "dependents": "Maybe",
        "tenure": 18,
        "contract": "Three year",
        "paperless_billing": "Fax",
        "payment_method": "Crypto",
        "monthly_charges": 65.5,
        "total_charges": 1179.0,
        "phone_service": "VoIP",
        "multiple_lines": "Three+",
        "internet_service": "Starlink",
        "online_security": "ThirdParty",
        "online_backup": "Cloud",
        "device_protection": "Warranty",
        "tech_support": "Premium",
        "streaming_tv": "4K",
        "streaming_movies": "IMAX"
    }])

    transformed = transform_new(unseen_row, artifacts)

    assert isinstance(transformed, pd.DataFrame)
    assert list(transformed.columns) == artifacts["feature_columns"]
    assert transformed.shape == (1, len(artifacts["feature_columns"]))
    assert transformed.isna().sum().sum() == 0


def test_raw_field_coverage(processed_dataset):
    """
    Verifies that the set of raw feature names recovered from the fitted preprocessor
    equals exactly the 19 raw input fields, and every transformed column maps to
    exactly one raw feature (longest-matching raw column name, not first prefix match).
    """
    _, _, artifacts = processed_dataset
    feature_cols = artifacts["feature_columns"]

    # Load raw columns (excluding target)
    df = load_data("data/churn.db")
    raw_cols = [c for c in df.columns if c != "churn"]
    assert len(raw_cols) == 19, f"Expected 19 raw fields, got {len(raw_cols)}: {raw_cols}"

    # Map each transformed column to its raw field using longest-match
    raw_from_transformed = set()
    mapping = {}
    for fname in feature_cols:
        if fname.startswith("num__"):
            raw_name = fname[5:]  # remove 'num__'
        elif fname.startswith("bin__"):
            raw_name = fname[5:]  # remove 'bin__'
        elif fname.startswith("cat__"):
            suffix = fname[5:]  # remove 'cat__'
            best_match = None
            best_len = 0
            for raw in raw_cols:
                if suffix.startswith(raw + "_") and len(raw) > best_len:
                    best_len = len(raw)
                    best_match = raw
            raw_name = best_match
            assert raw_name is not None, f"No raw match for transformed feature {fname}"
        else:
            raise AssertionError(f"Unknown prefix for transformed feature: {fname}")

        raw_from_transformed.add(raw_name)
        mapping[fname] = raw_name

    # All 19 raw fields must be covered
    missing = set(raw_cols) - raw_from_transformed
    assert not missing, f"Missing raw fields in transformed features: {sorted(missing)}"

    # Every transformed column maps to exactly one raw feature (already enforced by mapping dict)
    assert len(mapping) == len(feature_cols)