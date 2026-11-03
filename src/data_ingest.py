import os
import sqlite3

import pandas as pd

# Paths
DB_PATH = "data/churn.db"
CSV_PATH = "data/raw/Telecom_customer churn_100k.csv"


def create_database():
    """Loads the raw CSV into a single customers table, keeping every column."""
    df = pd.read_csv(CSV_PATH, low_memory=False)

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("customers", conn, index=False, if_exists="replace")
    conn.close()
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns into {DB_PATH}")


if __name__ == "__main__":
    # Delete old DB if exists (clean run)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    create_database()
