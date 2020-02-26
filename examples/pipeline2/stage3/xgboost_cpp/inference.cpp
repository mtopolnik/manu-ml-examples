#include "xgboost/c_api.h"
#include <iostream>
#include <stdio.h>
#include <cassert>
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <pthread.h>


pthread_mutex_t predict_mutex;

const float * predict(
	float* data,
	BoosterHandle* model, 
	int num_rows, 
	int num_cols) {

	std::cout << "Converting to DMatrix\n";
	DMatrixHandle dtest;
	XGDMatrixCreateFromMat(data, num_rows, num_cols, 0, &dtest);
  	
  	// predict
  	std::cout << "Calling predict.";
	bst_ulong out_len;
	const float * result;
	pthread_mutex_lock(&predict_mutex);
	XGBoosterPredict(*model, dtest, 0, 0, 0, &out_len, &result);  // This method is not thread safe.
	pthread_mutex_unlock(&predict_mutex);
	std::cout << "Finished predict.";
	assert(out_len == num_rows);
	return result;
}

void read_data(const char* filepath, std::vector<std::vector<float> >& result) {

	std::ifstream file(filepath);
    std::string line;
    
    while(std::getline(file,line)) {
	    std::stringstream lineStream(line);
	    std::string cell;
	    std::vector<float> row;
	    while(std::getline(lineStream, cell, ',')) {
	        row.push_back(::atof(cell.c_str()));
	    }
	    result.push_back(row);
    }
}


void printv(std::vector<float> input) {
	std::cout << "**** Number of elements in the row is : " << input.size() << "\n";
	for (std::vector<int>::size_type i = 0; i < input.size(); i++) {
		std::cout << input.at(i) << ' ';
	}
	std::cout << "\n";
}

void tomatrix(std::string input) {
	std::cout << "**** Number of elements in the row is : " << input.size() << "\n";
	for (std::vector<int>::size_type i = 0; i < input.size(); i++) {
		std::cout << input.at(i) << ' ';
	}
	std::cout << "\n";
}


int main() {
	const char *csv_path = "outputs.csv";
	const char *model_path = "models/xgboost.model";
	BoosterHandle booster;

	// LOAD MODEL
	std::cout << "Loading model.\n";
	XGBoosterCreate(NULL, 0, &booster);
	XGBoosterSetParam(booster, "seed", "0");  
	XGBoosterLoadModel(booster, model_path);
	std::cout << "Loaded model.\n";

	std::cout << "Reading data.\n";
	std::vector<std::vector<float> > data;
	read_data(csv_path, data);
	std::cout << "Finished reading data. Size: rows : " << data.size() << " Columns : " << data.at(0).size() << "\n";
	
	printf("Converting to array\n");
	int rows = data.size();
	int columns = data.at(0).size();
	float matrix[rows][columns];

	for (int i = 0; i < rows; i++) {
		std::copy(data.at(i).begin(), data.at(i).end(), matrix[i]);
	}
	
	const float * result = predict((float *)matrix, &booster, rows, columns);
	
	// PRINT RESULTS
	std::cout << "Predictions : [" << result[0];
	for (int i = 1; i < rows; i++) {
		std::cout << "," << result[i];
	}
	std::cout << "]\n";
	XGBoosterFree(booster);
	return 0;

}