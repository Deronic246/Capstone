from flask import Flask, render_template, request, jsonify
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.classification import LinearSVCModel
import os

from pyspark.sql.functions import *
import logging

from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol,TypeConverters
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.ml.param import Param, Params
from pyspark.sql.types import ArrayType, StringType
from nltk.stem import WordNetLemmatizer
from pyspark import keyword_only
from pyspark.conf import SparkConf
import nltk

app = Flask(__name__)

class Lemmatizer(Transformer, HasInputCol, HasOutputCol, DefaultParamsWritable, DefaultParamsReadable):
    input_col = Param(Params._dummy(), "input_col", "input column name.", typeConverter=TypeConverters.toString)
    output_col = Param(Params._dummy(), "output_col", "output column name.", typeConverter=TypeConverters.toString)
    @keyword_only
    def __init__(self, input_col: str = "input", output_col: str = "output"):
        super(Lemmatizer, self).__init__()
        self._setDefault(input_col=None, output_col=None)
        kwargs = self._input_kwargs
        self.set_params(**kwargs)
    
    @keyword_only
    def set_params(self, input_col: str = "input", output_col: str = "output"):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def get_input_col(self):
        return self.getOrDefault(self.input_col)

    def get_output_col(self):
        return self.getOrDefault(self.output_col)

    # Implement the transformation logic
    def _transform(self, dataset):
       
        input_column = self.get_input_col()
        output_column =  self.get_output_col()
        
        # Initialize the WordNetLemmatizer
        
        lemmatizer = WordNetLemmatizer()
        # Define the lemmatization function
        def lemmatize_tokens(tokens):
            pos_tags = nltk.pos_tag(tokens)
            lemmas = []
            for token in tokens:
                lemma = lemmatizer.lemmatize(token)
                lemmas.append(lemma)
            return lemmas

        # Register the UDF
        lemmatize_udf = udf(lemmatize_tokens, ArrayType(StringType()))

        # Apply transformation
        return dataset.withColumn(output_column, lemmatize_udf(input_column))

# Create a Spark session
spark = SparkSession.builder \
    .appName("newapp") \
    .getOrCreate()

# Get the directory containing the current script (app.py)
current_directory = os.path.dirname(os.path.abspath(__file__))

# Configure logging to a file
log_folder = os.path.join(current_directory, 'logging')
os.makedirs(log_folder, exist_ok=True)  # Create the logging folder if it doesn't exist

log_file = os.path.join(log_folder, 'app.log')
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
        review_dict = {"reviewText": review}
        #create dataframe using dictionary
        df = spark.createDataFrame([review_dict])
        #create user defined function to get sentiment score
        sentiment = udf(lambda x: TextBlob(x).sentiment[0])
        
        #register user defined function
        spark.udf.register("sentiment", sentiment)

        #generate sentiment score column
        df = df.withColumn('sentiment_score',sentiment('reviewText').cast('double'))
        #generate absolute sentiment score column
        df = df.withColumn('abs_sentiment_score', abs(spam_df['sentiment_score']))
        #generate review length column
        df = df.withColumn("review_text_length", length("reviewText"))
        #clean text
        newdf=spamCleanPipeline.transform(df)
        #generate features
        newdf=spamprepPipeline.transform(newdf)
        #predict if text is spam or ham based on features
        newdf=svmModel.transform(newdf)
        #get class of text
        sclass=int(newdf.first()["class"])
        return jsonify(sclass) 
    except Exception as e:
        # Log the error to the file
        app.logger.error('An error occurred: %s', e)
        return jsonify(3)
    
    

if __name__ == '__main__':
    app.run(debug=True)
