from flask import Flask, render_template, request, jsonify
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.classification import LinearSVCModel
from pyspark.ml.recommendation import ALSModel
import os
import re
from textblob import TextBlob
from pyspark.sql.functions import *
import logging
from transformers.lemmatizer import Lemmatizer
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol,TypeConverters
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.ml.param import Param, Params
from pyspark.sql.types import ArrayType, StringType,IntegerType,StructField,StructType
from nltk.stem import WordNetLemmatizer
from pyspark import keyword_only
from pyspark.conf import SparkConf
import nltk
from bs4 import BeautifulSoup
import psycopg2
import json
from sqlalchemy import create_engine
import pandas as pd

app = Flask(__name__)
app.config['DEBUG'] = True

nltk.download('wordnet')
nltk.download("averaged_perceptron_tagger")


# Create a Spark session
spark = SparkSession.builder \
    .appName("newapp").master("local[*]")\
     .config("spark.driver.memory", "15g") \
    .getOrCreate()

# Get the directory containing the current script (app.py)
current_directory = os.path.dirname(os.path.abspath(__file__))

# Configure logging to a file
log_folder = os.path.join(current_directory, 'logging')
os.makedirs(log_folder, exist_ok=True)  # Create the logging folder if it doesn't exist

log_file = os.path.join(log_folder, 'app.log')

with open(log_file, 'w'):
    pass

logging.basicConfig(filename=log_file, level=logging.ERROR)


#initialize models
kmeansModel=KMeansModel.load(os.path.join(current_directory, 'models', 'kmeansmodel'))
svmModel=LinearSVCModel.load(os.path.join(current_directory, 'models', 'svmmodel'))
#initialize pipelines
spamCleanPipeline=PipelineModel.load(os.path.join(current_directory, 'pipelines', 'spam_preproc_pipeline'))
spamprepPipeline=PipelineModel.load(os.path.join(current_directory, 'pipelines', 'data_prep_pipe'))

reviews_preproc_pipeline=PipelineModel.load(os.path.join(current_directory, 'pipelines', 'reviews_preproc_pipeline'))
clustering_pipeline=PipelineModel.load(os.path.join(current_directory, 'pipelines', 'clustering_pipeline1'))


db_config = {
    "host": "localhost",
    "user": "postgres",
    "password": "password",
    "database":"productdb"
}
engine = create_engine(
    "postgresql+psycopg2://{0}:{1}@{2}/{3}".format(db_config["user"],db_config["password"],db_config["host"],db_config["database"]))

#create user defined function to get sentiment score
sentiment = udf(lambda x: TextBlob(x).sentiment[0])

#register user defined function
spark.udf.register("sentiment", sentiment)

# Define a function for data cleaning
def clean_text(text):
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    # Remove Unicode characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Normalize text
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

# Define a UDF to calculate cosine similarity
def cosine_similarity(vector1, vector2):
    dot_product = float(vector1.dot(vector2))
    magnitude_product = float(vector1.norm(2) * vector2.norm(2))
    return dot_product / magnitude_product

# Register the clean_text function as a UDF (User-Defined Function)
clean_text_udf = udf(clean_text, StringType())

cosine_similarity_udf = udf(cosine_similarity)

@app.route('/')
@app.route('/home')
def home():
   
    return render_template('home.html')

@app.route('/reviews')
def reviews():
    return render_template('reviews.html')

@app.route('/ratings')
def ratings():
    
    return render_template('ratings.html')

@app.route('/populateOptions', methods=['POST'])
def populateOptions():
    rows_as_list=[]
    try:
      
        data = request.json
        
        connection = psycopg2.connect(**db_config)

        # Create a cursor to execute SQL queries
        cur = connection.cursor()

        # Formulate and execute the SELECT * query
        #query = f"SELECT distinct product_id, product_title FROM reviews  LOWER(product_title) LIKE %s"
        cur.execute("SELECT distinct product_id, product_title FROM reviews where LOWER(product_title) LIKE '%{0}%'".format(data["query"]))

        # Fetch all rows from the result set
        rows = cur.fetchall()

        # Convert rows to a list of lists
        rows_as_list = [list(row) for row in rows]

        # Close the cursor and connection
        cur.close()
        connection.close()
    except Exception as e:
            # Log the error to the file
            app.logger.error('\nAn error occurred: %s', e)
            return jsonify([])
    return jsonify(rows_as_list)

@app.route('/populateOptionsForRatings', methods=['POST'])
def populateOptionsForRatings():
    rows_as_list=[]
    try:
      
        data = request.json
        
        connection = psycopg2.connect(**db_config)

        # Create a cursor to execute SQL queries
        cur = connection.cursor()

        # Formulate and execute the SELECT * query
        cur.execute("SELECT distinct product_id, product_title FROM reviews where LOWER(product_title) LIKE '%{0}%'".format(data["query"]))

        # Fetch all rows from the result set
        rows = cur.fetchall()

        # Convert rows to a list of lists
        rows_as_list = [list(row) for row in rows]

        # Close the cursor and connection
        cur.close()
        connection.close()
    except Exception as e:
            # Log the error to the file
            app.logger.error('\nAn error occurred: %s', e)
            return jsonify([])
    return jsonify(rows_as_list)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # JSON data sent in the request
        
        review_dict = {"reviewText": data['review_text']}
        #create dataframe using dictionary
        df = spark.createDataFrame([review_dict])
     

        #generate sentiment score column
        df = df.withColumn('sentiment_score',sentiment('reviewText').cast('double'))
        #generate absolute sentiment score column
        df = df.withColumn('abs_sentiment_score', abs(df['sentiment_score']))
        #generate review length column
        df = df.withColumn("review_text_length", length("reviewText"))

        

        # Create a new column "review_body_clean" by applying the clean_text UDF to "review_body"
        df = df.withColumn("reviewText_clean", clean_text_udf("reviewText"))
        
        #clean text
        newdf=spamCleanPipeline.transform(df)
        #generate features
        newdf=spamprepPipeline.transform(newdf)
        #predict if text is spam or ham based on features
        newdf=svmModel.transform(newdf)
        #get class of text
      
        #app.logger.info("Columns of DataFrame: %s", ''.join(newdf.columns))
        sclass=int(newdf.first()["prediction"])
        return jsonify(sclass) 
    except Exception as e:
        # Log the error to the file
        app.logger.error('\nAn error occurred: %s', e)
        return jsonify(3)
    
@app.route('/recommendProductsByReview', methods=['POST'])
def recommendProductsByReview():
    try:
        data = request.json  # JSON data sent in the request       
      
     
        schema = StructType([StructField("id", IntegerType(), True), StructField("customer_id", StringType(), True)\
                             ,StructField("product_id", StringType(), True),StructField("product_category", StringType(), True)\
                             ,StructField("product_title", StringType(), True),StructField("review_id", StringType(), True)\
                              ,StructField("review_body", StringType(), True),StructField("star_rating", IntegerType(), True)
                             ,StructField("customer_id_index", IntegerType(), True),StructField("product_id_index", IntegerType(), True)])
        pdf = pd.read_sql('select * from reviews', engine)
        
        # Convert Pandas dataframe to spark DataFrame
        df = spark.createDataFrame(pdf,schema)

        #generate sentiment score column
        df = df.withColumn('sentiment_score',sentiment('review_body').cast('double'))
        
        #generate absolute sentiment score column
        df = df.withColumn('abs_sentiment_score', abs(df['sentiment_score']))
        
        #generate review length column
        df = df.withColumn("review_text_length", length("review_body"))

        

        # Create a new column "review_body_clean" by applying the clean_text UDF to "review_body"
        df = df.withColumn("review_body_clean", clean_text_udf("review_body"))

        #clean text
        newdf=reviews_preproc_pipeline.transform(df)
        grouped_df = newdf.groupBy("product_id","product_title","product_category").agg(
            avg("abs_sentiment_score").alias("avg_abs_sentiment"),\
            avg("review_text_length").alias("avg_review_length"),\
            round(avg("star_rating"),0).alias("avg_star_rating"),\
            collect_list("lemmas").alias("combined_tokens")
        )
        newdf = grouped_df.withColumn("combined_tokens", flatten(col("combined_tokens")))
        
        #generate features
        newdf=clustering_pipeline.transform(newdf)

        newdf=kmeansModel.transform(newdf)
        
        target_product_cluster = newdf.filter(col("product_id")==data["id"]).first()["prediction"]

        product_cluster_data=newdf\
        .select("prediction","product_id","features","product_title","avg_star_rating","product_category")\
        .filter(col("prediction")==target_product_cluster)

        product_data=newdf.select("prediction","product_id","features","product_title","avg_star_rating","product_category")\
        .filter((col("prediction")==target_product_cluster) & (col("product_id")==data["id"])).limit(1)

        # Cross-join normalized features with itself to get all pairwise combinations
        cross_joined_data = product_data.alias("a").crossJoin(product_cluster_data.alias("b"))

        # Calculate cosine similarity and select relevant columns
        cosine_similarity_df = cross_joined_data.select(
            "a.product_id",
            "b.product_id",
            cosine_similarity_udf("a.features", "b.features").alias("cosine_similarity")
        )
        # Filter out self-pairs (where product_id1 = product_id2)
        cosine_similarity_df = cosine_similarity_df.filter(col("a.product_id") != col("b.product_id"))\
        .orderBy(col("cosine_similarity").desc())

        top5prodIDs=cosine_similarity_df.limit(5).select(col("b.product_id")).withColumn("product_id",trim(col("product_id"))).withColumnRenamed("avg_star_rating", "star_rating")

        distinctProducts=product_cluster_data.select("product_id","product_title","avg_star_rating","product_category").distinct()\
        .withColumn("product_id",trim(col("product_id"))).withColumnRenamed("avg_star_rating", "star_rating")

        product_ids_list = top5prodIDs.rdd.flatMap(lambda x: x).collect()
        product_ids_list.insert(0, data["id"])

        filtered_product_details_df = distinctProducts.filter(col("product_id").isin(product_ids_list))
        
        json_rdd = filtered_product_details_df.toJSON()
        json_records = json_rdd.collect()

        #Convert the list of JSON records to a list of dictionaries
        records = [json.loads(record) for record in json_records]
        
        return jsonify(records) 
    except Exception as e:
        # Log the error to the file
        app.logger.error('\nAn error occurred: %s', e)
        return jsonify([])

    
@app.route('/recommendProductsByRating', methods=['POST'])
def recommendProductsByRating():
    connection = psycopg2.connect(**db_config)

        # Create a cursor to execute SQL queries
    cur = connection.cursor()
    try:
        data = request.json  # JSON data sent in the request
        query="select distinct customer_id_index from reviews where product_id='{0}' LIMIT 1;".format(data["id"])
        
  
        
        pdf = pd.read_sql(query, engine)

        
        # Convert Pandas dataframe to spark DataFrame
        df = spark.createDataFrame(pdf)

      
       
     
        model=ALSModel.load(os.path.join(current_directory, 'models', 'alsmodel'))

        user_set = df.withColumn("customer_id_index", df["customer_id_index"].cast(IntegerType()))
        column_names = ["product_id", "product_title", "star_rating", "product_category"]

        recommendations=model.recommendForUserSubset(user_set,5)
        if recommendations.count()==0:
            app.logger.error('\nNo recommendations available')
            product_id=data["id"]
            # SQL query to retrieve the first record that matches the product_id
            query = f"SELECT product_id,product_title,star_rating,product_category FROM reviews WHERE product_id='{product_id}' LIMIT 1;"

            cur.execute(query)
            results = cur.fetchall()
            list_of_dicts = [dict(zip(column_names, row)) for row in results]
            return jsonify(list_of_dicts)
        else:
            recs=recommendations.withColumn("itemAndRating",explode(recommendations.recommendations))\
            .select("customer_id_index","itemAndRating.*")


            recs=recs.withColumn("product_id_index", recs["product_id_index"].cast(IntegerType()))
  
             # Convert DataFrame to a JSON string
            json_strings = recs.toJSON().collect()

            # Combine JSON strings into a single string
            json_string = "[" + ",".join(json_strings) + "]"

            
            # Using a loop
            formatted_string = ""
            for column_name, data_type in recs.dtypes:
                formatted_string += f"('{column_name}', '{data_type}'), "

            formatted_string = formatted_string.rstrip(', ')  # Remove the trailing comma and space
            

            product_id_list=recs.select("product_id_index").rdd.flatMap(lambda x: x).collect()

           

            
       
            # SQL query to retrieve the first record for each product
            query1 = 'SELECT product_id,product_title,star_rating,product_category FROM reviews  WHERE product_id_index IN {}'.format(tuple(product_id_list))

            

            product_id=data["id"]
            # SQL query to retrieve the first record that matches the product_id
            query2 = f"SELECT product_id,product_title,star_rating,product_category FROM reviews WHERE product_id='{product_id}' LIMIT 1;"
            
        
            # Formulate and execute the SELECT * query
            cur.execute(query1)

            # Fetch all rows from the result set
            results1 = cur.fetchall()

            cur.execute(query2)
            results2 = cur.fetchall()
            

            finalList=results2+results1
            
            list_of_dicts = [dict(zip(column_names, row)) for row in finalList]
            return jsonify(list_of_dicts)
            
            return jsonify(finalList) 
    except Exception as e:
        # Log the error to the file
        app.logger.error('\nAn error occurred: %s', e)
        return jsonify([])
    
    finally:
        # Close the cursor and the connection
        if cur:
            cur.close()
        if connection:
            connection.close()
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000)
