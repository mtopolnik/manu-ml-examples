#import pandas as pd
import os
import pickle
import datetime
import json
import yaml
import sys

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DateType, IntegerType, DoubleType, BooleanType
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.session import SparkSession

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import sklearn
from sklearn import metrics
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from common import *
from sklearn.preprocessing import OneHotEncoder
import xgboost

SAMPLE = 500
SPARK_NTREES = 10

'''
# Set up environment
sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)
spark = SparkSession(sc)
'''

spark = SparkSession.builder.master('local[{}]'.format(spark_threads)).appName('local-testing-pyspark-context').getOrCreate()


def spark_one_hot_encoding(df, columns):
	stages = []
	output_columns = []
	for col in columns:
		op_col = col + "_feature_vec"
		output_columns.append(op_col)
		stringIndexer = StringIndexer(inputCol = col, outputCol = col + 'Index', handleInvalid = 'keep')
		encoder = OneHotEncoderEstimator(dropLast = True, inputCols=[stringIndexer.getOutputCol()], outputCols=[op_col])
		stages += [stringIndexer, encoder]
	return stages, output_columns


'''
 This is the inference code. Nothing too fancy. 
 We wil simply convert to ML format and build a
 model.

'''
def train(df, models_folder):
	print("Building Spark model.")

	df = spark.createDataFrame(df, schema = StructType(get_spark_df_schema(df)))

	x_cols = list(df.columns)
	x_cols.remove(target_column)

	# all the columns of type string
	categorical = [c for c in x_cols if dict(df.dtypes)[c].startswith('s')]

	# remove those from x
	x_cols = [col for col in x_cols if col not in categorical]

	stages, output_columns = spark_one_hot_encoding(df, categorical)

	# add the one hot incoded columns to x
	x_cols.extend(output_columns)

	df2 = df.withColumn(target_column, F.col(target_column).cast(StringType()))

	stages.append(VectorAssembler(inputCols=x_cols, outputCol="features", handleInvalid = 'keep'))
	stages.append(StringIndexer(inputCol = target_column, outputCol = 'label', handleInvalid = 'keep'))
	stages.append(RandomForestClassifier(labelCol = 'label', featuresCol = "features", numTrees=SPARK_NTREES))
	pipeline = Pipeline(stages = stages)
	model = pipeline.fit(df2)
	results = model.transform(df2)
	# FIXME: lets print something here like auc just for sanity
	model.save(models_folder + '/' + ml_model_name)
	print("Finished running training pipeline. Outputs are in ", models_folder)
	return model


if __name__ == "__main__":
	
	if len(sys.argv) != 2:
		print("Usage: <models folder>")
		exit(1)

	models_folder = sys.argv[1]

	if os.path.exists(models_folder):
		raise ValueError("Output path exists.")

	os.mkdir(models_folder)
	df = pd.read_json(stage2_outputs)
	print("Read data from stage 2. Size: ", len(df))
	if SAMPLE: 
		print("Sampling the data.")
		df = df.sample(SAMPLE)
	print("Size of training dataset will be: ", len(df))


	train(df, models_folder)



