import pandas as pd
import sklearn
import os
import pickle
import datetime


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from . import util

datasets_dir = '../../datasets'
filename = os.path.join(datasets_dir, 'income.data.txt')
target_column = 'yearly-income'

def read_csv(filename, has_header = True, sep = ','):
	df = pd.read_csv(filename, header = 0 if has_header else None)
	df.columns = [col.strip() for col in df.columns]
	return df


def train_model_simple():

	df = read_csv(filename)
	df_train = df.drop(target_column, axis = 1)
	labels = df[target_column]

	# split into train and test
	x_train, x_test, y_train, y_test = train_test_split(df_train, labels, test_size=0.2)

	# create the encoder object (this is an example of an ML specific transform for categorical data)
	enc_obj = OneHotEncoder(sparse = False, handle_unknown = 'ignore')

	encoder = enc_obj.fit(x_train)
	x_train_encoded = encoder.transform(x_train)

	# we are not going to tune any hyper-parameters
	rf = RandomForestClassifier(n_estimators = 10, n_jobs = -1)  # -1 uses all cores...
	model = rf.fit(x_train_encoded, y_train)

	# get training accuracy
	training_predictions = model.predict_proba(x_train_encoded).transpose()[1] # get predicstions for label 1
	print("Auc on training data is ", util.auc(y_train, training_predictions, model.classes_[1]))

	# now accuracy on test set
	x_test_encoded = encoder.transform(x_test)
	test_predictions = model.predict_proba(x_test_encoded).transpose()[1] # get predicstions for label 1
	print("Auc on training data is ", util.auc(y_test, test_predictions, model.classes_[1]))

	print("Dumping models as pickle file.")
	with open('rf.model', 'wb') as f:
		pickle.dump(model, f)

	with open('onehot.model', 'wb') as f:
		pickle.dump(encoder, f)



def train_model_pipeline():
	df = read_csv(filename)
	df_train = df.drop(target_column, axis = 1)
	labels = df[target_column]

	# split into train and test
	x_train, x_test, y_train, y_test = train_test_split(df_train, labels, test_size=0.2)

	# create the encoder object (this is an example of an ML specific transform for categorical data)
	enc_obj = OneHotEncoder(sparse = False, handle_unknown = 'ignore')
	normalizer = Normalizer()
	rf = RandomForestClassifier(n_estimators = 10, n_jobs = -1)  # -1 uses all cores...
	
	pipeline = Pipeline([('onehot', enc_obj), ('norm', normalizer)])
	pipeline_model = pipeline.fit(x_train, y_train)
	x_train_encoded = pipeline_model.transform(x_train)
	model = rf.fit(x_train_encoded, y_train)


	# get training accuracy
	training_predictions = model.predict_proba(x_train_encoded).transpose()[1] # get predicstions for label 1
	print("Auc on training data is ", util.auc(y_train, training_predictions, model.classes_[1]))

	# now accuracy on test set
	x_test_encoded = pipeline_model.transform(x_test)
	test_predictions = model.predict_proba(x_test_encoded).transpose()[1] # get predicstions for label 1
	print("Auc on training data is ", util.auc(y_test, test_predictions, model.classes_[1]))

	print("Dumping models as pickle file.")
	with open('rf_pipeline.model', 'wb') as f:
		pickle.dump(model, f)

	with open('pipeline.model', 'wb') as f:
		pickle.dump(pipeline_model, f)


if __name__ == "__main__":
	train_model_simple()
	train_model_pipeline()
