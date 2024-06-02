This project aims to predict the presence of heart disease in patients using various machine learning models. The dataset used is the "Heart Disease" dataset from the Cleveland Clinic Foundation.

Project Structure
Data Pre-processing

Load and explore the dataset
Plot distributions of variables
Check correlations among independent variables
Gaussian test to check if the density plot is Gaussian
Normalize variables using Min-Max Normalization
Stratified sampling
Oversampling
Feature Selection

Feature importance analysis
Build dataset with selected features
Model Building and Evaluation

Decision Tree
Artificial Neural Network (ANN)
K-Nearest Neighbors (KNN)
Getting Started
Prerequisites
R version 4.0 or higher
R packages: caTools, DataExplorer, ROSE, mlr, rpart, rpart.plot, party, partykit, neuralnet, caret, e1071, ROCR
Data
The dataset used in this project is heart_cleveland_upload.csv. Ensure this file is placed in the appropriate directory before running the scripts.

Installation
Install the required R packages using the following commands:

R
Copy code
install.packages("caTools")
install.packages("DataExplorer")
install.packages("ROSE")
install.packages("mlr")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("party")
install.packages("partykit")
install.packages("neuralnet")
install.packages("caret")
install.packages("e1071")
install.packages("ROCR")
Running the Project
Data Pre-processing

Load and explore the dataset
Plot variable distributions
Normalize the data
Perform stratified sampling and oversampling
Feature Selection

Analyze feature importance
Create a dataset with selected features
Model Building and Evaluation

Build and evaluate Decision Tree models
Build and evaluate ANN models
Build and evaluate KNN models
Code Overview
Data Pre-processing
The script loads the dataset, plots distributions, checks correlations, normalizes variables, and performs stratified sampling and oversampling.

Feature Selection
Feature importance is analyzed, and a dataset with selected features is created.

Model Building and Evaluation
Various models are built and evaluated, including Decision Trees, ANNs, and KNNs. The models' performance is assessed using confusion matrices, ROC curves, and AUC.

Example Plots
ROC Curve:
Used to evaluate the performance of classification models.
Example command to plot ROC curve:
R
Copy code
library(ROCR)
pred1 = prediction(pred_model, test_set2$condition)
perf = ROCR::performance(pred1, "tpr", "fpr")
plot(perf, colorize = TRUE, main = "ROC curve", ylab = "Sensitivity", xlab = "Specificity", print.cutoffs.at = seq(0, 1, 0.3), text.adj = c(-0.2, 1.7))
Results
The project uses multiple machine learning algorithms to predict heart disease.
Models are evaluated based on accuracy, confusion matrices, and ROC curves.
The best-performing model is determined based on the evaluation metrics.
