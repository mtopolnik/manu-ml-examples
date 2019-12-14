# Modular Data Science Project
## Introduction
This is a more realistic data science project that a better data scientist would create.
It uses a public lending club dataset.
The models are trying to predict whether a loan will default.


## Stage 1 - Feature Engineering
NLP Processing and dimensionality reduction.


## Stage 2 - Join/Enrichment/Munging
Joins the input data to a second file with a foreign key.
This is a quite common need.

## Stage 3 - ML
### XGBoost 
It is a GBM model  from xgboost, a Python/C++ model - Gradient Boosting Machines model. Very popular in financial services due to accuracy and has an advantage that it can be significantly smaller than random forest.

Training and Infernce - both need stage 1 and stage 2

### Spark Random Forest Model

Training and Inference using Spark and Spark ML.
Uses Python APIs but could use Java APIs

## Jet Implementation Idea

Add Stage 1 and Stage 2 to a data prep job which then outputs to Stage 3. Stage 2 should leverage data locality

Training and Inference need all three stages - there is a different version one for train and one for inference in each, but the steps are still needed.







