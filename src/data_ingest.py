import sqlite3
import pandas as pd
import os

# Paths
DB_PATH = "data/churn.db"
SCHEMA_PATH = "src/sql_schema.sql"
CSV_PATH = "data/raw/telco_customer_churn.csv"


def create_database():
    """Creates SQLite DB and tables using schema file."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Execute SQL schema
    with open(SCHEMA_PATH, "r") as f:
        cursor.executescript(f.read())

    conn.commit()
    conn.close()
    print("Database and tables created successfully!")


def insert_data():
    """Loads CSV and inserts records into normalized SQL tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Load dataset
    df = pd.read_csv(CSV_PATH)

    # Fix numeric formatting
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df["Total Charges"].fillna(0, inplace=True)

    # Insert into tables using executemany for efficiency
    cursor.execute("DELETE FROM customers")
    cursor.execute("DELETE FROM accounts")
    cursor.execute("DELETE FROM services")

    # Prepare and insert CUSTOMERS data
    customers_data = df[['CustomerID', 'Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Tenure Months', 'Churn Label']].copy()
    customers_data['Senior Citizen'] = customers_data['Senior Citizen'].map({'Yes': 1, 'No': 0})
    customers_tuples = [tuple(row) for row in customers_data.to_numpy()]
    cursor.executemany("""
        INSERT INTO customers (customer_id, gender, senior_citizen, partner, dependents, tenure, churn)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, customers_tuples)

    # Prepare and insert ACCOUNTS data
    accounts_data = df[['CustomerID', 'Contract', 'Paperless Billing', 'Payment Method', 'Monthly Charges', 'Total Charges']]
    accounts_tuples = [tuple(row) for row in accounts_data.to_numpy()]
    cursor.executemany("""
        INSERT INTO accounts (customer_id, contract, paperless_billing, payment_method,
                            monthly_charges, total_charges)
        VALUES (?, ?, ?, ?, ?, ?)
    """, accounts_tuples)

    # Prepare and insert SERVICES data
    services_data = df[['CustomerID', 'Phone Service', 'Multiple Lines', 'Internet Service',
                       'Online Security', 'Online Backup', 'Device Protection',
                       'Tech Support', 'Streaming TV', 'Streaming Movies']]
    services_tuples = [tuple(row) for row in services_data.to_numpy()]
    cursor.executemany("""
        INSERT INTO services (customer_id, phone_service, multiple_lines, internet_service,
                            online_security, online_backup, device_protection,
                            tech_support, streaming_tv, streaming_movies)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, services_tuples)

    conn.commit()
    conn.close()
    print("Data inserted successfully!")


if __name__ == "__main__":
    # Delete old DB if exists (clean run)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    create_database()
    insert_data()
