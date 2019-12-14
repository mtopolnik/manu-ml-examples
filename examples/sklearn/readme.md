Basic Data Science Project 
One Python model to train the model and dump it.
Inputs are where is the data, config yaml file with model parameters etc
Spits out yaml, json
Does some analysis
Testing files that loads the model:
- example_1_inference_batch.py - Loads model and makes predictions on the supplied test data file and emits a JSON output
- example_1_inference_streaming.py - Sets up a Flask REST API server, loads model and waits for data on port 5000. Makes predictions one at a time on each request emits a JSON output
