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
from common import *

#################################################
# Adjustable parameters
batch_size = 100
sample_size = None
run_mode = 'batch' # 'batch' or 'streaming'
outputs = 'outputs.csv'
#################################################


if len(sys.argv) != 2:
	print("Usage: <models folder> ")
	exit(1)

models_folder = sys.argv[1]

# Load the one hot model
with open(models_folder + '/' + onehot_model_name,  'rb') as f:
    ohe = pickle.load(f)

def read_data():
	print("Reading data")
	df = pd.read_json(stage2_outputs)
	if sample_size:
		df = df.sample(sample_size)

	print("Size of test data is : ", len(df))
	return df

def main():
	print("Run mode is {}. Sample Size is : {}, batch size is {}".format(run_mode, sample_size, batch_size))
	
	print("Starting run...")
	df = read_data()

	new_df = ohe.transform(df)
	new_df.to_csv(outputs, index = False, header = False)
	print("Writing pre-processed test data to ", outputs)
	


if __name__ == "__main__":
	main()

