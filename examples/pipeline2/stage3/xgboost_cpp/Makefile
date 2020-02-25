# build an executable named myprog from myprog.c
all: inference.cpp 
	g++ -O3 -I../../xgboost/rabit/include -I../../xgboost/include -o predict inference.cpp ../../xgboost/lib/libxgboost.a ../../xgboost/rabit/lib/librabit.a ../../xgboost/dmlc-core/libdmlc.a

clean: 
	$(RM) predict