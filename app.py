from flask import Flask, render_template, request, jsonify
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.classification import LinearSVCModel
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
from pyspark.sql.types import ArrayType, StringType
from nltk.stem import WordNetLemmatizer
from pyspark import keyword_only
from pyspark.conf import SparkConf
import nltk
from bs4 import BeautifulSoup
app = Flask(__name__)

nltk.download('wordnet')
nltk.download("averaged_perceptron_tagger")


# Create a Spark session
spark = SparkSession.builder \
    .appName("newapp").master("local[*]")\
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

product_data={"product_id":"1",'category':"Laptops","title":"HP Notebook","rating":2,\
                   "imageurl":\
                   "https://mdbcdn.b-cdn.net/img/Photos/Horizontal/E-commerce/Products/4.webp",\
                  "similar_products":[{"product_id":"2",'category':"Laptops","title":"HP Notebook","rating":5,\
                   "imageurl":\
                   "https://mdbcdn.b-cdn.net/img/Photos/Horizontal/E-commerce/Products/4.webp",\
                  "similar_products":[],"reviews":[]},{"product_id":"3",'category':"Laptops","title":"HP Notebook","rating":3,\
                   "imageurl":\
                   "https://mdbcdn.b-cdn.net/img/Photos/Horizontal/E-commerce/Products/7.webp",\
                  "similar_products":[],"reviews":[]},{"product_id":"5",'category':"Laptops","title":"HP Notebook","rating":4,\
                   "imageurl":\
                   "https://mdbcdn.b-cdn.net/img/Photos/Horizontal/E-commerce/Products/5.webp",\
                  "similar_products":[],"reviews":[]},{"product_id":"8",'category':"gdfgd","title":"HP Notebook","rating":5,\
                   "imageurl":\
                   "https://mdbcdn.b-cdn.net/img/Photos/Horizontal/E-commerce/Products/5.webp",\
                  "similar_products":[],"reviews":[]},{"product_id":"11",'category':"Laptops","title":"HP Notebook","rating":1,\
                   "imageurl":\
                   "https://mdbcdn.b-cdn.net/img/Photos/Horizontal/E-commerce/Products/5.webp",\
                  "similar_products":[],"reviews":[]}],"reviews":[{"customer_id":"34232323","verified_purchase":"yes","review_date":"1997-11-20","review_body":"fdfsdfsddfs fsdfdfs dfsdfsdf","review_type":"Ham","helpful_votes":45}]}



@app.route('/')
@app.route('/home')
def home():
   
    return render_template('home.html',data=product_data)

@app.route('/reviews')
def reviews():
    # In a real app, you would fetch reviews from a database or API
    # For this example, we'll use some dummy data
    
    return render_template('reviews.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # JSON data sent in the request
        
        review_dict = {"reviewText": data['review_text']}
        #create dataframe using dictionary
        df = spark.createDataFrame([review_dict])
        #create user defined function to get sentiment score
        sentiment = udf(lambda x: TextBlob(x).sentiment[0])
        
        #register user defined function
        spark.udf.register("sentiment", sentiment)

        #generate sentiment score column
        df = df.withColumn('sentiment_score',sentiment('reviewText').cast('double'))
        #generate absolute sentiment score column
        df = df.withColumn('abs_sentiment_score', abs(df['sentiment_score']))
        #generate review length column
        df = df.withColumn("review_text_length", length("reviewText"))

        # Register the clean_text function as a UDF (User-Defined Function)
        clean_text_udf = udf(clean_text, StringType())

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
