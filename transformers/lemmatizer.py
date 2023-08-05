from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.ml.param import Param, Params
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
from nltk.stem import WordNetLemmatizer
from pyspark import keyword_only
from pyspark.conf import SparkConf
import nltk

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