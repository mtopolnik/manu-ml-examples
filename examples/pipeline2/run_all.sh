#!/bin/bash

# train stage 1

echo "***** Running stage 1"
cd stage1
rm -rf models
python train.py ../../../datasets models
ls models

# create the outputs in stage 1 for stage 2
# default is batch mode
python inference.py ../../../datasets models

# completed stage1
echo "***** Running stage 2"
cd ..
cd stage2
# create outputs in stage 2 for stage 3
# there is no need to train as this is just a join
python inference.py ../../../datasets

# completed stage2

# xgboost
echo "***** Running stage 3: XGBoost"
cd ..

cd stage3/xgboost
rm -rf models
python train.py models
ls models

python inference.py models
head outputs.json


# spark
echo "***** Running stage 3: Spark"
cd ../spark
rm -rf models
python train.py models
ls models/

# inference
python inference.py models
head outputs.json


# xgboost cpp
echo "***** Running stage 3: XGBoost CPP"
cd ../xgboost_cpp
rm -rf models
python train.py models
ls models/  # should contain onehot encoded model and xgboost model

# inference
# Data preprocessing is in python, would have been cumbersome to write in C++.
# load the python models and dump a numeric only matrix that feeds into the ML model
# This basically does reads the one-hot-encoding model dumped into the 'models'
# folder above during the training phase, and then applies it to the data output
# in stage 2 to dump outputs.csv which is a numeric only matrix 
python preprocess_test_data.py models  
head -n1 outputs.csv

# predict on outputs.csv using the C++ Xgboost inteface. Model is the same as dumped by train.py
make clean all
./predict


