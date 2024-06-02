# AMLproject
#R_language
rm(list=ls())
###1. data pre-processing
dataset=read.csv("C:\\Users\\aslan\\OneDrive\\Documents\\AML\\heart_cleveland_upload.csv")
str(dataset)

#1.1 plot distributions
barplot(table(dataset$age),col=c("aquamarine1"))
barplot(table(dataset$sex),col=c("aquamarine1","brown2"))
barplot(table(dataset$cp),col=c("aquamarine1","brown2"))
barplot(table(dataset$trestbps),col=c("aquamarine1"))
barplot(table(dataset$chol),col=c("aquamarine1"))
barplot(table(dataset$fbs),col=c("aquamarine1","brown2"))
barplot(table(dataset$restecg),col=c("aquamarine1","brown2"))
barplot(table(dataset$thalach),col=c("aquamarine1"))
barplot(table(dataset$exang),col=c("aquamarine1","brown2"))
barplot(table(dataset$oldpeak),col=c("aquamarine1"))
barplot(table(dataset$slope),col=c("aquamarine1","brown2"))
barplot(table(dataset$ca),col=c("aquamarine1","brown2"))
barplot(table(dataset$thal),col=c("aquamarine1","brown2"))
barplot(table(dataset$condition),col=c("aquamarine1","brown2"))

#1.2 check correlation among independent variables
corrTable=cor(dataset[-14])
library('DataExplorer')
plot_correlation(dataset[-14],type='continuous', cor_args = list("use" = "pairwise.complete.obs"))

#1.3 Gaussian test - Check if the density plot is Gaussian
plot_density(dataset)

#1.4 normalization variables by Min- Max Normalization
normalize=function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
dataset=as.data.frame(lapply(dataset, normalize))
str(dataset)
plot_density(dataset)

#1.5 reattach the target variable
#dataset$condition=factor(dataset$condition)
condition=factor(dataset$condition)
dataset1=dataset[-14]
dataset=cbind(dataset1,condition)
str(dataset)

#1.6 Stratified sampling
#build original dataset 
library(caTools)
set.seed(123)
split = sample.split(dataset$condition, SplitRatio = 0.7)
training_set1 = subset(dataset, split == TRUE)
test_set1 = subset(dataset, split == FALSE)
table(training_set1$condition)
table(dataset$condition)
table(test_set1$condition)
prop.table(table(dataset$condition))
barplot(table(dataset$condition),col=c("aquamarine1","brown2"))
barplot(table(training_set1$condition),col=c("aquamarine1","brown2"))
barplot(table(test_set1$condition),col=c("aquamarine1","brown2"))

#1.7 Oversampling
table(training_set1$condition)
prop.table(table(training_set1$condition))
barplot(table(training_set1$condition),col=c("aquamarine1","brown2"))
library("ROSE")
data_balaced_over=ovun.sample(condition~.,data=training_set1,method="over",N=1040)$data
data_balaced_over=ovun.sample(condition~.,data=data_balaced_over,method="over",N=1856)$data
training_set3=data_balaced_over
table(training_set3$condition)
barplot(table(training_set3$condition),col=c("aquamarine1","brown2"))
test_set3=test_set1

#1.8 feature
library(mlr)
dataset=read.csv("C:\\Users\\aslan\\OneDrive\\Documents\\AML\\heart_cleveland_upload.csv")
train.task=makeRegrTask(data = dataset, target = "condition")
var_imp2=generateFilterValuesData(train.task, method = "linear.correlation")
plotFilterValues(var_imp2, feat.type.cols = TRUE, n.show = 30)
library(corrplot)
corrplot(cor(dataset), order = "hclust")

#1.9 build dataset with feature selection
training_set2=training_set1[-4:-7]
test_set2=test_set1[-4:-7]
str(training_set2)
str(test_set2)
#models
###1. decision tree
library(RCurl)
library(rpart)
library(rpart.plot)
library(party)
library(caTools)
#1.1.1 build tree with Gini index 
tree_gini = rpart(condition~ ., data=training_set2)
rpart.plot(tree_gini, extra = 101, nn = TRUE)
#1.1.2 build tree with entropy information
tree_entropy=rpart(condition~ ., data=training_set2, method="class", parms=list(split="information"))
rpart.plot(tree_entropy, extra = 101, nn = TRUE)
#1.1.3 bulid tree This code generates the tree with training data
tree_with_params = rpart(condition~ ., data=training_set2, method="class", minsplit = 1, minbucket = 10, cp = -1)
rpart.plot(tree_with_params, extra = 101, nn = TRUE)
#1.1.4 build tree using Party Package
library(partykit)
tree_party = ctree(condition~ ., data=training_set2)
plot(tree_party)

#1.2.1 predict using training set
Predict_gini= predict(tree_gini, training_set2, type = "class")
Predict_entropy= predict(tree_entropy, training_set2, type = "class")
Predict_with_params = predict(tree_with_params, training_set2, type = "class")
predict_party=predict(tree_party,training_set2)
#1.2.2 accuracy of validation
cm1=table(Predict_gini,training_set1$condition)
confusionMatrix(cm1)
cm2=table(Predict_entropy,training_set2$condition)
confusionMatrix(cm2)
cm3=table(Predict_with_params,training_set2$condition)
confusionMatrix(cm3)
cm4=table(predict_party,training_set2$condition)
confusionMatrix(cm4)


#1.3.1 predict using validation data (type=default)
Predict_gini= predict(tree_gini, test_set2, type = "class")
Predict_entropy= predict(tree_entropy, test_set2, type = "class")
Predict_with_params = predict(tree_with_params, test_set2, type = "class")
predict_party=predict(tree_party,test_set2)
#1.3.2 accuracy of validation
cm1=table(Predict_gini,test_set2$condition)
confusionMatrix(cm1)
cm2=table(Predict_entropy,test_set2$condition)
confusionMatrix(cm2)
cm3=table(Predict_with_params,test_set2$condition)
confusionMatrix(cm3)
cm4=table(predict_party,test_set2$condition)
confusionMatrix(cm4)

#1.4 ROC curve and AUC of best model
library(ROCR)
pred_model=as.numeric(Predict_with_params)
pred1=prediction(pred_model,test_set2$condition)
perf=ROCR::performance(pred1,"tpr","fpr")
plot(perf,colorize=T,main="ROC curve",ylab="Sensitivity",xlab="Specificity",
     print.cutoffs.at=seq(0,1,0.3),text.adj= c(-0.2,1.7))
auc=as.numeric(ROCR::performance(pred1,"auc")@y.values)
auc

###2 ANN
library (neuralnet)
#2.1 compare different hidden layers 
set.seed(123)
nn_hidden10=neuralnet(condition~., data = training_set2,
                      hidden = c(6,6),
                      err.fct = "ce",
                      linear.output = FALSE)
plot(nn_hidden10)

#2.2 predict and evaluate on training set
#2.2.1 predict
output_training_nn10=compute(nn_hidden10, training_set2[,-10])
pred_training_nn10=output_training_nn10$net.result[,2]
pred_training_nn10=ifelse(pred_training_nn10 > 0.5, 1, 0)
#2.2.1 accuracy
cm_training_nn10=table(pred_training_nn10, training_set2$condition)
confusionMatrix(cm_training_nn10)

#2.3 predict and evaluate on test set
#2.3.1 predict
output_test_nn10=compute(nn_hidden10, test_set2[,-10])
#output_test_nn10$net.result
pred_test_nn10=output_test_nn10$net.result[,2]
pred_test_nn10=ifelse(pred_test_nn10 > 0.5, 1, 0)
#2.3.2 accuracy
cm_test_nn10=table(pred_test_nn10,test_set2$condition)
confusionMatrix(cm_test_nn10)


#2.4 different algorithms
nn_sag=neuralnet(condition~., data = training_set2,
                 hidden =10,
                 err.fct = "ce",
                 algorithm = "sag",
                 linear.output = FALSE)
nn_slr=neuralnet(condition~., data = training_set2,
                 hidden =10,
                 err.fct = "ce",
                 algorithm = "slr",
                 linear.output = FALSE)
nn_rpropplus=neuralnet(condition~., data = training_set2,
                       hidden =10,
                       err.fct = "ce",
                       algorithm = "rprop+",
                       linear.output = FALSE)
nn_rprop=neuralnet(condition~., data = training_set2,
                   hidden =10,
                   err.fct = "ce",
                   algorithm = "rprop-",
                   linear.output = FALSE)

#2.5 predict and evaluate on training set
#2.5.1 predict
output_training_nn10=compute(nn_sag, training_set2[,-10])
pred_training_nn10=output_training_nn10$net.result[,2]
pred_training_nn10=ifelse(pred_training_nn10 > 0.5, 1, 0)
#2.5.2 accuracy
cm_training_nn10=table(pred_training_nn10, training_set1$condition)
confusionMatrix(cm_training_nn10)

#2.6 predict and evaluate on test set
#2.6.1 predict
output_test_nn10=compute(nn_slr, test_set2[,-10])
#output_test_nn10$net.result
pred_test_nn10=output_test_nn10$net.result[,2]
pred_test_nn10=ifelse(pred_test_nn10 > 0.5, 1, 0)
#2.6.2 accuracy
cm_test_nn10=table(pred_test_nn10,test_set1$condition)
confusionMatrix(cm_test_nn10)

#2.7 ROC curve and AUC of best model
library(ROCR)
pred_model=as.factor(pred_test_nn10)
pred_model=as.numeric(pred_model)
pred1=prediction(pred_model,test_set2$condition)
perf=ROCR::performance(pred1,"tpr","fpr")
plot(perf,colorize=T,main="ROC curve",ylab="Sensitivity",xlab="Specificity",
     print.cutoffs.at=seq(0,1,0.3),text.adj= c(-0.2,1.7))
auc=as.numeric(ROCR::performance(pred1,"auc")@y.values)
auc

###3 KNN
#3.1 Defining the training controls for multiple models
Trian_Control=trainControl(
  method = "cv",
  number = 5,
  savePredictions = 'final',
  classProbs = T)

#3.2 build KNN model
training_set2$condition=factor(training_set2$condition,levels=c(0,1),labels=c("N","Y"))
test_set2$condition=factor(test_set2$condition,levels=c(0,1),labels=c("N","Y"))

model_knn=train(condition~., data = training_set2, method='knn', trControl=Trian_Control, tuneLength=1)

#3.3 predict using training_set
pred_knn=predict(object = model_knn, training_set2)
confusionMatrix(training_set2$condition,pred_knn)

#3.4 predict using test set

pred_knn=predict(object = model_knn, test_set1)
confusionMatrix(test_set1$condition,pred_knn)

#3.5 ROC curve and AUC of best model
library(ROCR)
pred_model=as.numeric(pred_knn)
pred1=prediction(pred_model,test_set1$condition)
perf=ROCR::performance(pred1,"tpr","fpr")
plot(perf,colorize=T,main="ROC curve",ylab="Sensitivity",xlab="Specificity",
     print.cutoffs.at=seq(0,1,0.3),text.adj= c(-0.2,1.7))
auc=as.numeric(ROCR::performance(pred1,"auc")@y.values)
auc
