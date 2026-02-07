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

    # Insert into tables row-by-row
    for _, row in df.iterrows():

        # Insert into CUSTOMERS table
        cursor.execute("""
            INSERT INTO customers (
                customer_id, gender, senior_citizen, partner, dependents, tenure, churn
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            row['CustomerID'],
            row['Gender'],
             1 if row['Senior Citizen'] == "Yes" else 0,
            row['Partner'],
            row['Dependents'],
            row['Tenure Months'],
            row['Churn Label']
        ))

        # Insert into ACCOUNTS table
        cursor.execute("""
            INSERT INTO accounts (
                customer_id, contract, paperless_billing, payment_method,
                monthly_charges, total_charges
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            row['CustomerID'],
            row['Contract'],
            row['Paperless Billing'],
            row['Payment Method'],
            row['Monthly Charges'],
            row['Total Charges']
        ))

        # Insert into SERVICES table
        cursor.execute("""
            INSERT INTO services (
                customer_id, phone_service, multiple_lines, internet_service,
                online_security, online_backup, device_protection,
                tech_support, streaming_tv, streaming_movies
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row['CustomerID'],
            row['Phone Service'],
            row['Multiple Lines'],
            row['Internet Service'],
            row['Online Security'],
            row['Online Backup'],
            row['Device Protection'],
            row['Tech Support'],
            row['Streaming TV'],
            row['Streaming Movies']
        ))

    conn.commit()
    conn.close()
    print("Data inserted successfully!")


if __name__ == "__main__":
    # Delete old DB if exists (clean run)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    create_database()
    insert_data()
