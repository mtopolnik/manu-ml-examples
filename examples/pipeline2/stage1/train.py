#import pandas as pd
import os
import pickle
import sys
import pandas as pd
import json

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from common import *

SAMPLE = 10000

# This stage uses Sk Learn to create
# a structured and reduced dimensional
# output from the text basec col: emp_title
def _train(df, models_folder):
	n = 256
	# TFIDF for text data
	df.loc[df['emp_title'].isnull(), 'emp_title'] = "NA"
	
	with open(models_folder + '/' + input_dtypes, "w") as f:
		json.dump({x:str(y) for x, y in dict(df.dtypes).items()}, f, indent = 4)

	vectorizer = TfidfVectorizer()
	model = vectorizer.fit(df['emp_title'])
	tfidf_op = model.transform(df['emp_title'])
	
	# reduce dimensionality to n
	svd = TruncatedSVD(n)
	svd_model = svd.fit(tfidf_op)
	d = svd_model.transform(tfidf_op)
	
	emp_title_df = pd.DataFrame(d, columns = ["emp_title_svd" + str(i) for i in range(n)])
	df = df.drop('emp_title', axis = 1)
	return pd.concat([df, emp_title_df], axis = 1), model, svd_model

def train(df, models_folder):


	# Stage 1: Tf Idf and SVD using sklearn
	print("Running stage 1 ")
	df1, tf_model, svd_model = _train(df, models_folder)
	
	with open(models_folder + '/' + tfidf_model_name,  'wb') as f:
		pickle.dump(tf_model, f)
	with open(models_folder + '/' + svd_model_name,  'wb') as f:
		pickle.dump(svd_model, f)

	print("Finished running stage 1. Outputs are in ", models_folder)

	return df1


if __name__ == "__main__":
	
	if len(sys.argv) != 3:
		print("Usage: <input folder> <models folder>")
		exit(1)

	inputs_folder = sys.argv[1]
	models_folder = sys.argv[2]

	if os.path.exists(models_folder):
		raise ValueError("Models folder already exists.")

	os.mkdir(models_folder)

	print("Reading data.")
	df = pd.read_csv(inputs_folder + '/' + 'streaming.csv', low_memory = False)

	if SAMPLE:
		df = df.sample(SAMPLE)
	
	print("Dataset size: ", len(df))
	df = train(df, models_folder)

	


