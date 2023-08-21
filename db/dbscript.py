import json
import psycopg2
from psycopg2 import sql
import pandas as pd
import os
import logging
from pyspark.sql import SparkSession,DataFrame
from pyspark.sql.functions import *
from pyspark.conf import SparkConf
# PostgreSQL database configuration
db_config = {
    "host": "localhost",
    "user": "postgres",
    "password": "password"
}

# Create a SparkSession
spark = SparkSession.builder.appName("amazonapp").getOrCreate()

# Get the parent directory of the current script's directory
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# JSON file containing review data
#json_file_path = os.path.join(parent_directory, 'dataset', 'reviews10k.json')

# Open the JSON file and load data
#df=pd.read_json(json_file_path)
df=spark.read.option("header", "true")\
.json(os.path.join(parent_directory, 'dataset', 'reviews.json')).sample(0.04,67).toPandas()
#logging.info("testing")

# Configure logging to a file
log_folder = os.path.join(parent_directory, 'logging')
os.makedirs(log_folder, exist_ok=True)  # Create the logging folder if it doesn't exist

log_file = os.path.join(log_folder, 'dbscript.log')

with open(log_file, 'w'):
    pass

 # Configure the logging system 
logging.basicConfig(filename =log_file, level = logging.ERROR) 
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
            product_category VARCHAR(100),
            product_title Text,
            review_id VARCHAR(100),
            review_body TEXT,
            star_rating INTEGER,
            customer_id_index INTEGER,
            product_id_index INTEGER
        )
    """
    cursor.execute(create_table_query)


    # Iterate through the DataFrame using iterrows and insert rows into the table
    for index, row in df.iterrows():
        query = f"INSERT INTO reviews(customer_id, product_id,product_category,product_title,review_body,star_rating,customer_id_index,product_id_index) VALUES ( %s, %s, %s, %s, %s, %s, %s, %s)"
        values = (row["customer_id"], row["product_id"], row["product_category"],row["product_title"],  row["review_body"], row["avg_star_rating"],row["customer_id_index"],row["product_id_index"])
        cursor.execute(query, values)
        connection.commit()

    


except Exception as e:
    logging.error("An exception occurred: %s", str(e))
    spark.stop()

finally:
    if connection is not None:
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed.")
        spark.stop()
