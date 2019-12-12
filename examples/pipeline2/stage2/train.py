import os
import pickle
import sys
import pandas as pd
import json

def train(df, inputs_folder):
	print("Reading metadata.")
	metadata = pd.read_csv(inputs_folder + "/metadata.csv", low_memory=False)
	return df.merge(metadata, on = 'id')


# doesnt do anything. just for completeness.
if __name__ == "__main__":
	
	if len(sys.argv) != 2:
		print("Usage: <input folder>")
		exit(1)

	inputs_folder = sys.argv[1]
	print("Reading streaming.")
	df = pd.read_csv(inputs_folder + "/streaming.csv", low_memory=False)
	df = train(df, inputs_folder)



