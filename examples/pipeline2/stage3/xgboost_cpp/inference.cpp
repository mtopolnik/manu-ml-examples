#include "xgboost/c_api.h"
//#include "xgboost/data.h"

#include <iostream>
#include <stdio.h>

#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>


void predict(
	DMatrixHandle* dtest,
	BoosterHandle* model, 
	int num_rows, 
	int num_cols, 
	const float* result) {
	//dtest = (DMatrix*) dtest;
	//std::cout << "Info:\n" << dtest->Info() << "\n";
	bst_ulong rows, columns;
	XGDMatrixNumRow(*dtest, &rows);
	std::cout << "Num rows in dataset: " << rows << "\n";
	XGDMatrixNumCol(*dtest, &columns);
	std::cout << "Num cols in dataset: " << columns << "\n";
	
	std::cout << "Converting to DMatrix\n";

	//std::cout << dtest[0];
	//XGDMatrixCreateFromMat(rows,
    //                     num_rows, num_cols, NULL, &dtest);
  	// predict
	bst_ulong out_len;
	std::cout << "Here 1\n";
	
	//const float *f;
	XGBoosterPredict(*model, *dtest, 0, 0, &out_len, &result);

	std::cout << "here 2\n";
	// assert(out_len == num_rows);
	std::cout << result[0] << std::endl;
	for (int i = 0; i < rows; i++) {
		std::cout << "Result " << i << " " << result[i] << " \n";
	}

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
	const char *csv_path = "outputs.csv?format=csv";
	const char *model_path = "models/xgboost.model";
	BoosterHandle booster;

	// LOAD MODEL

	// create booster handle first
	XGBoosterCreate(NULL, 0, &booster);

	// by default, the seed will be set 0
	XGBoosterSetParam(booster, "seed", "0");
      
	// load model
	XGBoosterLoadModel(booster, model_path);
	printf("Loaded model.\n");

	printf("Loading data from x++ model.\n");
	DMatrixHandle dmatrix;
	int ret_val = XGDMatrixCreateFromFile(csv_path, 0, &dmatrix);
	std::cout << "Matrix load status: " << ret_val << "\n";
	// LOAD CSV DATA
	std::vector<std::vector<float> > data;
	std::cout << "Reading data.\n";
	//read_data(csv_path, data);
	std::cout << "Finished data.\n";
	
	//float matrix[data.size()][data.at(0).size()];
	//printf("Converting to matrix\n");
	
	//for (int i = 0; i < data.size(); i++) {
	//	std::copy(data.at(i).begin(), data.at(i).end(), matrix[i]);
	//}
	
	const float * result;
	
	printf("Calling predict.\n");

	predict(&dmatrix, &booster, 0,0, result);
	
	printf("Finished predicting \n");

	XGBoosterFree(booster);
	return 0;

}