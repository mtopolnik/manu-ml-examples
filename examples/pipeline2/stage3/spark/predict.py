import pandas as pd
import json
from common import *

def infer(rows, model, spark):
	"""
		rows: a list of dictionaries each dictionary represents a row of data [{'col1':'value1',...}]
	"""
	df = pd.DataFrame(rows).drop(target_column, axis = 1)
	df = spark.createDataFrame(df, schema = StructType(get_spark_df_schema(df)))
	result = model.transform(df)
	return [x['probability'][0] for x in result.select('probability').collect()]
	