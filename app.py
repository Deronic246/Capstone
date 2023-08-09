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
from pyspark.sql.types import ArrayType, StringType,IntegerType
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
kmeansModel=KMeansModel.load(os.path.join(current_directory, 'models', 'kmeans'))
svmModel=LinearSVCModel.load(os.path.join(current_directory, 'models', 'svm'))
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
        cur.execute("SELECT distinct product_id, product_title FROM ratings where LOWER(product_title) LIKE '%{0}%'".format(data["query"]))

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
      
     
        
        pdf = pd.read_sql('select * from reviews', engine)
        
        # Convert Pandas dataframe to spark DataFrame
        df = spark.createDataFrame(pdf)

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
        #generate features
        newdf=clustering_pipeline.transform(newdf)

        newdf=kmeansModel.transform(newdf)
        
        target_product_cluster = newdf.filter(col("product_id")==data["id"]).first()["prediction"]

        product_cluster_data=newdf\
        .select("prediction","product_id","normalized_features","product_title","star_rating","product_category")\
        .filter(col("prediction")==target_product_cluster)

        product_data=newdf.select("prediction","product_id","normalized_features","product_title","star_rating","product_category")\
        .filter((col("prediction")==target_product_cluster) & (col("product_id")==data["id"])).limit(1)

        # Cross-join normalized features with itself to get all pairwise combinations
        cross_joined_data = product_data.alias("a").crossJoin(product_cluster_data.alias("b"))

        # Calculate cosine similarity and select relevant columns
        cosine_similarity_df = cross_joined_data.select(
            "a.product_id",
            "b.product_id",
            cosine_similarity_udf("a.normalized_features", "b.normalized_features").alias("cosine_similarity")
        )
        # Filter out self-pairs (where product_id1 = product_id2)
        cosine_similarity_df = cosine_similarity_df.filter(col("a.product_id") != col("b.product_id"))\
        .orderBy(col("cosine_similarity").desc())

        top5prodIDs=cosine_similarity_df.limit(5).select(col("b.product_id")).withColumn("product_id",trim(col("product_id")))

        distinctProducts=product_cluster_data.select("product_id","product_title","star_rating","product_category").distinct()\
        .withColumn("product_id",trim(col("product_id")))

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
        query="select distinct customer_id_index from ratings where product_id='{0}' LIMIT 1;".format(data["id"])
        
        app.logger.error('\nQuery: {0}'.format(query))
        
        pdf = pd.read_sql(query, engine)
        app.logger.error('\Pandas count: {0}'.format(pdf.shape[0]))
        app.logger.error('\Pandas data: {0}'.format(pdf.at[0, "customer_id_index"]))
        app.logger.error('\nIs Dataset empty: {0}'.format(pdf.empty))
        
        # Convert Pandas dataframe to spark DataFrame
        df = spark.createDataFrame(pdf)

      
       
     
        model=ALSModel.load(os.path.join(current_directory, 'models', 'alsmodel'))

        user_set = df.withColumn("customer_id_index", df["customer_id_index"].cast(IntegerType()))

        recommendations=model.recommendForUserSubset(user_set,5)
        if recommendations.count()==0:
            app.logger.error('\nNo recommendations available')
            product_id=data["id"]
            # SQL query to retrieve the first record that matches the product_id
            query = f"SELECT product_id,product_title,star_rating,product_category FROM ratings WHERE product_id='{product_id}' LIMIT 1;"

            cur.execute(query)
            results = cur.fetchall()
            return jsonify(results)
        else:
            app.logger.error('\n Count before: {0}'.format(recommendations.count()))
            recs=recommendations.withColumn("itemAndRating",explode(recommendations.recommendations))\
            .select("customer_id_index","itemAndRating.*")

            app.logger.error('\n Count customer_id_index: {0}'.format(recs.select("customer_id_index").count()))
           
            app.logger.error('\n Count before: {0}'.format(recs.select("product_id_index").count()))
            recs=recs.withColumn("product_id_index", recs["product_id_index"].cast(IntegerType()))
            app.logger.error('\n Count before: {0}'.format(recs.select("product_id_index").count()))
            app.logger.error('\n Before flatmap. Columns: {0}'.format(''.join(recs.columns)))

             # Convert DataFrame to a JSON string
            json_strings = recs.toJSON().collect()

            # Combine JSON strings into a single string
            json_string = "[" + ",".join(json_strings) + "]"

            app.logger.error('\n json: {0}'.format(json_string))
            
            # Using a loop
            formatted_string = ""
            for column_name, data_type in recs.dtypes:
                formatted_string += f"('{column_name}', '{data_type}'), "

            formatted_string = formatted_string.rstrip(', ')  # Remove the trailing comma and space
            app.logger.error('\n dtypes: {0}'.format(formatted_string))
            product_id_list=recs.select("product_id_index").rdd.flatMap(lambda x: x).collect()

           

            
            app.logger.error('\n After flatmap. Columns: {0}'.format(''.join(recs.columns)))
            # Generate a comma-separated string of product IDs for the query
            #product_ids_str = ",".join(map(str, product_ids))
            
            #app.logger.error('\Products: {0}'.format(product_ids_str))
            # SQL query to retrieve the first record for each product
            query1 = 'SELECT product_id,product_title,star_rating,product_category FROM ratings WHERE product_id_index IN (' \
           + (',?' * len(product_id_list))[1:] + ');'

            app.logger.error('\nQuery1: {0}'.format(query1))

            product_id=data["id"]
            # SQL query to retrieve the first record that matches the product_id
            query2 = f"SELECT product_id,product_title,star_rating,product_category FROM ratings WHERE product_id='{product_id}' LIMIT 1;"
            app.logger.error('\nQuery2: {0}'.format(query2))

        
            # Formulate and execute the SELECT * query
            cur.execute(query1,product_id_list)

            # Fetch all rows from the result set
            results1 = cur.fetchall()

            cur.execute(query2)
            results2 = cur.fetchall()
            
            # Convert rows to a list of lists
            rows_as_list1 = [list(row) for row in results1]
            rows_as_list2 = [list(row) for row in results2]

            finalList=rows_as_list2+rows_as_list1
            

            
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
    app.run(host='0.0.0.0', port=8080)
