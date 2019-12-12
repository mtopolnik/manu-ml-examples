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
