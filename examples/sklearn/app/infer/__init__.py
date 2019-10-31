import os
import logging
import sys
from flask_restful import Api, Resource, reqparse, fields, marshal
from flask import Flask, jsonify, abort, make_response, send_file
from flask import request
from flask_httpauth import HTTPBasicAuth

app = Flask(__name__)
app.secret_key = os.urandom(24)

api = Api(app)


def load_pipeline_model(transforms_file, model_file):


	return transfoms, model


transforms, model = load_pipeline_model()

class Predict(Resource):
    
    def post(self):
        data = parser.parse_args()

api.add_resource(resources.Predict, '/predict')





