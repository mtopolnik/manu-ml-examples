import pandas as pd
import numpy as np
import sklearn
import os
import pickle
import datetime
import json
import yaml
import sys
import requests
import threading
import time
import numpy as np

from flask_restful import Api, Resource, reqparse, fields, marshal
from flask import Flask, jsonify
from flask import request

import predict

#################################################
# Adjustable parameters
batch_size = 100
sample_size = 1000
run_mode = 'batch' # 'batch' or 'streaming'
outputs = 'outputs.json'
stage1_output = '../stage1/outputs.json'
#################################################


if len(sys.argv) != 2:
	print("Usage: <input folder> ")
	exit(1)

inputs_folder = sys.argv[1]

metadata = pd.read_csv(inputs_folder + "/metadata.csv", low_memory=False)

def read_data():
	print("Reading data...")
	df = pd.read_json(stage1_output)
	if sample_size:
		df = df.sample(sample_size)
	print("Size of test data is : ", len(df))
	return df
	

def run_batch():
	print("Starting run...")
	df = read_data()
	data = df.to_dict(orient  = "records")
	results = []
	start = datetime.datetime.now()

	for rows in np.array_split(data, len(data)/batch_size):
		results.extend(predict.infer(list(rows), metadata))
		print("Processed {} rows".format(len(results)))
	end = datetime.datetime.now()
	
	print("Time take to process {} rows is {}".format(len(data), end - start))
	assert(len(results) == len(data))
	print("Writing results to ", outputs)
	with open(outputs, 'w') as f:
		json.dump(results, f, indent = 4)
	print("First result is: " , results[0:1])


class PredictionsAPI(Resource):

	def __init__(self):
		super(PredictionsAPI, self).__init__()
		self.reqparse = reqparse.RequestParser()
		self.reqparse.add_argument('data',
		type=dict, required=True,
		help='No data provided', location='json')

	def post(self):
		args = self.reqparse.parse_args()
		data = args['data']
		return predict.infer(list(data.values()), metadata)


def streaming_test():
	sleep_time = 3
	print("***** In streaming test. Will sleep for {} seconds.".format(sleep_time))
	time.sleep(sleep_time)
	print("***** Awake. Starting to send data to endpoint...")
	print("Reading data")
	df = read_data()
	data = json.loads(df.to_json(orient  = "records"))
	start = datetime.datetime.now()
	results = []

	for rows in np.array_split(data, len(data)/batch_size):
		d = {i: rows[i] for i in range(len(rows))}
		results.extend(requests.post("http://localhost:5000/predict", json = {'data': d}).json())
		print("Processed {} rows".format(len(results)))
	
	end = datetime.datetime.now()
	assert(len(results) == len(data))
	print("Time take to process {} rows is {}".format(len(data), end - start))
	print("First result is: " , results[0:1])
	print("Finished streaming test.")


def main():

	print("Run mode is {}. Sample Size is : {}, batch size is {}".format(run_mode, sample_size, batch_size))
	
	if run_mode == 'batch':
		run_batch()
	else:
		app = Flask(__name__)
		app.secret_key = os.urandom(24)
		api = Api(app)
		api.add_resource(PredictionsAPI, '/predict')
		t = threading.Thread(target=streaming_test)
		t.daemon = True
		t.start()
		app.run(debug=False)



if __name__ == "__main__":
	main()

