import json
import psycopg2
from psycopg2 import sql
import pandas as pd
import os

# PostgreSQL database configuration
db_config = {
    "host": "localhost",
    "user": "postgres",
    "password": "password"
}
# Get the parent directory of the current script's directory
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# JSON file containing review data
json_file_path = os.path.join(parent_directory, 'dataset', 'reviews10k.json')

# Open the JSON file and load data
df=pd.read_json(json_file_path)

# Connect to the PostgreSQL server (without specifying dbname)
connection=None
try:
    connection = psycopg2.connect(**db_config)
    connection.autocommit = True  # Autocommit mode for database creation
    cursor = connection.cursor()
    cursor.execute(sql.SQL("DROP DATABASE IF EXISTS productdb"))
    # Create the database if it doesn't exist
    cursor.execute(sql.SQL("CREATE DATABASE productdb"))

except psycopg2.errors.DuplicateDatabase:
    pass  # Database already exists, continue without creating

finally:
    if connection is not None:
        cursor.close()
        connection.close()

db_config = {
    "host": "localhost",
    "user": "postgres",
    "password": "password",
    "database":"productdb"
}
# Connect to the created database
try:
    connection = psycopg2.connect(**db_config)
    cursor = connection.cursor()
    cursor.execute(sql.SQL("Drop Table if exists reviews"))
    # Create the reviews table if it doesn't exist
    create_table_query = """
        CREATE TABLE IF NOT EXISTS reviews (
            id SERIAL PRIMARY KEY,
            customer_id VARCHAR(100),
            product_id VARCHAR(100),
            product_title Text,
            review_id VARCHAR(100),
            review_body TEXT,
            star_rating INTEGER
        )
    """
    cursor.execute(create_table_query)


    # Iterate through the DataFrame using iterrows and insert rows into the table
    for index, row in df.iterrows():
        query = f"INSERT INTO reviews(customer_id, product_id,product_title,review_id,review_body,star_rating) VALUES ( %s, %s, %s, %s, %s, %s)"
        values = (row["customer_id"], row["product_id"], row["product_title"], row["review_id"], row["review_body"], row["star_rating"])
        cursor.execute(query, values)
        connection.commit()



except psycopg2.Error as err:
    print("Error:", err)

finally:
    if connection is not None:
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed.")
