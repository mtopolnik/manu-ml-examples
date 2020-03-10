#!/bin/bash
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
mkdir build
cd build
# the default compiler on my system is fine.
# Needs to libomp on OSX "brew install libomp"
cmake -DBUILD_STATIC_LIB=ON ..
make -j2
cd ..
cd python-package/
python setup.py install
