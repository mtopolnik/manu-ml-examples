import pandas as pd
import json 


def infer(rows, metadata):
	"""
		rows: a list of dictionaries each dictionary represents a row of data [{'col1':'value1',...}]
	"""

	df = pd.DataFrame(rows)
	return json.loads(df.merge(metadata, on = 'id').to_json(orient = 'records')) 


