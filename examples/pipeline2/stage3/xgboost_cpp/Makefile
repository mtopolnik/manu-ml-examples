# build an executable named myprog from myprog.c
all: inference.cpp 
	g++ -O3 -I../../xgboost/rabit/include -I../../xgboost/include -lomp -o predict inference.cpp ../../xgboost/lib/libxgboost.a ../../xgboost/build/rabit/librabit.a ../../xgboost/build/dmlc-core/libdmlc.a
clean: 
	$(RM) predict