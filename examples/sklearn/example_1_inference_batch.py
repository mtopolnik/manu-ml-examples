import pandas as pd
import sklearn
import os
import pickle
import datetime
import json
import yaml
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
import util


target_column = 'yearly-income'

def predict(df, inputs_folder, ai_folder):

	# Seperate the target column
	labels = None
	if target_column in df.columns:
		labels = df[target_column]
		labels_dict = util.read_coll(ai_folder + '/labels_dict.json')
		labels = labels.apply(lambda x: labels_dict[x])
		df = df.drop(target_column, axis = 1)

	cols = util.read_coll(ai_folder + '/input_column_names.json')

	if sorted(cols) != sorted(list(df.columns)):
		print(cols)
		print(list(df.columns))
		raise ValueError("The columns do not match...")

	df = df[cols]   # ensure the same order of data

	# Join with occupation stats
	occupation_stats = pd.read_csv(inputs_folder + '/occupation_stats.csv')

	# join with the training data
	df = df.merge(occupation_stats, how = 'inner', on = 'occupation')

	# select only the feature selection columns
	feature_selected_columns = util.read_coll(ai_folder + '/selected_features.txt')
	df = df[feature_selected_columns]

	# add feature engineered columns
	inter_1 = pd.read_csv(ai_folder + '/education_mean.csv')
	df = df.merge(inter_1, on = 'education')

	# now may we read some control variables from a yaml file
	with open(inputs_folder + '/ai-config.yml', 'r') as stream:
	    config = yaml.load(stream)
	df['age'] = config['age_multiplier'] * df['age']

	hours_info  = util.read_coll(ai_folder + '/hours_info.json')
	df['hours-per-week'] = df['hours-per-week'] - hours_info['min']
	df['hours-per-week'] = df['hours-per-week'] / (hours_info['max'] - hours_info['min'])

	# create the encoder object (this is an example of an ML specific transform for categorical data)
	with open(ai_folder + '/onehot.model', 'rb') as f:
		encoder = pickle.load(f)

	x_test_encoded = encoder.transform(df)

	# we are not going to tune any hyper-parameters
	with open(ai_folder + '/rf.model', 'rb') as f:
		model = pickle.load(f)

	# get training accuracy
	test_predictions = model.predict_proba(x_test_encoded).transpose()[1] # get predictions for label 1

	if labels is not None:
		print("Auc on testing data is ", util.auc(labels, test_predictions, model.classes_[1]))

	print("Predictions distribution.")
	print(pd.Series(test_predictions).describe())

	predictions_file = 'predictions_{}.json'.format(util.get_timestamp())
	print("Dumping predictions file to ", predictions_file)
	util.write_coll(list(test_predictions), predictions_file)


if __name__ == "__main__":
	if len(sys.argv) != 4:
		print("Usage: <testing file> <input folder> <ai folder>")
		exit(1)

	filename = sys.argv[1]
	inputs_folder = sys.argv[2]
	ai_folder = sys.argv[3]

	df = util.read_csv(filename)
	predict(df, inputs_folder, ai_folder)

