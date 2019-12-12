import pandas as pd
import json 

def infer(rows, types, tfidf_model, svd_model):
	"""
		rows: a list of dictionaries each dictionary represents a row of data [{'col1':'value1',...}]
	"""
	df = pd.DataFrame(rows)
	df.loc[df['emp_title'].isnull(), 'emp_title'] = "NA"
	op = tfidf_model.transform(df['emp_title'])
	op = svd_model.transform(op)
	op = pd.DataFrame(op, columns = ["emp_title_svd" + str(i) for i in range(len(op[0]))])
	df = df.drop('emp_title', axis = 1)
	df = pd.concat([df, op], axis = 1)
	return json.loads(df.to_json(orient = 'records'))


