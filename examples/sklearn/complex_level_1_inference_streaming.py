import pandas as pd
import sklearn
import os
import pickle
import datetime
import json
import yaml
import sys
import util

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from flask_restful import Api, Resource, reqparse, fields, marshal
from flask import Flask, jsonify
from flask import request


target_column = 'yearly-income'

if len(sys.argv) != 3:
	print("Usage: <input folder> <ai folder>")
	exit(1)

inputs_folder = sys.argv[1]
ai_folder = sys.argv[2]


labels_dict = util.read_coll(ai_folder + '/labels_dict.json')
inv_labels_dict = {v: k for k, v in labels_dict.items()}
cols = util.read_coll(ai_folder + '/input_column_names.json')
# Join with occupation stats
occupation_stats = pd.read_csv(inputs_folder + '/occupation_stats.csv')
# select only he feature selection columns
feature_selected_columns = util.read_coll(ai_folder + '/selected_features.txt')
inter_1 = pd.read_csv(ai_folder + '/education_mean.csv')
	
with open(inputs_folder + '/ai-config.yml', 'r') as stream:
    config = yaml.load(stream)

hours_info  = util.read_coll(ai_folder + '/hours_info.json')


# create the encoder object (this is an example of an ML specific transform for categorical data)
with open(ai_folder + '/onehot.model', 'rb') as f:
	encoder = pickle.load(f)

# we are not going to tune any hyper-parameters
with open(ai_folder + '/rf.model', 'rb') as f:
	model = pickle.load(f)

def predict(row_dict):
	df = pd.DataFrame([row_dict])
	# join with the training data
	df = df.merge(occupation_stats, how = 'inner', on = 'occupation')
	df = df[feature_selected_columns]

	# add feature engineered columns
	df = df.merge(inter_1, on = 'education')

	# now may we read some control variables from a yaml file
	df['age'] = config['age_multiplier'] * df['age']
	df['hours-per-week'] = df['hours-per-week'] - hours_info['min']
	df['hours-per-week'] = df['hours-per-week'] / (hours_info['max'] - hours_info['min'])
	x_test_encoded = encoder.transform(df)

	# get training accuracy
	predicted_probability = model.predict_proba(x_test_encoded).transpose()[1][0] # get predicstions for label 1
	predicted_label =  model.predict(x_test_encoded).transpose()[0]
	x = {
		'probability': float(predicted_probability),
		'label': inv_labels_dict[int(predicted_label)]
	}
	return x


app = Flask(__name__)
app.secret_key = os.urandom(24)
api = Api(app)


class PredictionsAPI(Resource):
    
    def __init__(self):
        super(PredictionsAPI, self).__init__()
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('data', required=True, 
                                    help='No data provided',
                                    location='json')
    def post(self):
        
        args = self.reqparse.parse_args()
        print(type(eval(args['data'])))
        return predict(eval(args['data']))


api.add_resource(PredictionsAPI, '/predict')


def main():
	app.run(debug=True)

if __name__ == "__main__":
	main()

# curl -i -H "Content-Type: application/json" -X POST -d '{"data": {"age": 39, "workclass": "State-gov", "fnlwgt": 77516, "education": "Bachelors", "education-num": 13, "marital-status": "Never-married", "occupation": "Adm-clerical", "relationship": "Not-in-family", "race": "White", "sex": "Male", "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States", "yearly-income": "<=50K"} }' http://localhost:5000/predict






