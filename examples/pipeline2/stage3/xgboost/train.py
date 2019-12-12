#import pandas as pd
import os
import pickle
import datetime
import json
import yaml
import sys

import pandas as pd

from sklearn.pipeline import Pipeline
import sklearn
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
import xgboost
from common import *

SAMPLE = None
XGBOOST_NTREES = 10

def train(df, models_folder):
	
	print("Building XgBoost Model")
	
	pipeline = MyXgboost(
		OneHotEncoder(sparse = False, handle_unknown = 'ignore'), 
		xgboost.XGBClassifier(n_estimators = XGBOOST_NTREES))
	
	pipeline.fit(df)
	
	# let's print the accuracy for some basic sanity
	results = pipeline.transform(df.drop(target_column, axis = 1))
	fpr, tpr, thresholds = sklearn.metrics.roc_curve(df[target_column], results)
	print("Auc on training data is ", sklearn.metrics.auc(fpr, tpr))
	print(pd.DataFrame(results).describe())

	# write
	with open(models_folder + '/' + ml_model_name,  'wb') as f:
		pickle.dump(pipeline, f)

	print("Finished running training pipeline. Outputs are in ", models_folder)


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



