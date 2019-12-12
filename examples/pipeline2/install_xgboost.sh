#!/bin/bash
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
mkdir build
cd build
# the default compiler on my system is fine.
cmake -DUSE_OPENMP=OFF ..
cd ..
sed -i .bak 's/USE_OPENMP = 1/USE_OPENMP = 0/' make/config.mk
make -j2
cd python-package/
python setup.py install
