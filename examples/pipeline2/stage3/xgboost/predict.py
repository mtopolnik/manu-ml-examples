import pandas as pd
import json

def infer(rows, model):
	"""
		rows: a list of dictionaries each dictionary represents a row of data [{'col1':'value1',...}]
	"""
	df = pd.DataFrame(rows, dtype = float)
	result = model.transform(df)
	return list(result.astype(float))
	