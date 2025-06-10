import pandas as pd
import sqlite3
import os

# Path to your CSV file
csv_file = 'yc_companies.csv'

# Path for the SQLite database
db_file = 'yc_companies.db'

# Check if CSV file exists
if not os.path.exists(csv_file):
    print(f"Error: CSV file '{csv_file}' not found.")
    exit(1)

# Create a connection to the SQLite database
# If the file doesn't exist, it will be created
conn = sqlite3.connect(db_file)
print(f"Connected to SQLite database: {db_file}")

try:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    print(f"Read {len(df)} rows from {csv_file}")
    
    # Display the first few rows to verify data
    print("\nSample data:")
    print(df.head(3))
    
    # Write the DataFrame to a SQLite table
    # If the table already exists, it will be replaced
    df.to_sql('companies', conn, if_exists='replace', index=False)
    print(f"\nSuccessfully imported data into 'companies' table")
    
    # Verify the data was imported correctly
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM companies")
    count = cursor.fetchone()[0]
    print(f"Total rows in database: {count}")
    
    # Show table schema
    cursor.execute("PRAGMA table_info(companies)")
    columns = cursor.fetchall()
    print("\nTable schema:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")

except Exception as e:
    print(f"Error: {e}")

finally:
    # Close the connection
    conn.close()
    print("\nDatabase connection closed")
