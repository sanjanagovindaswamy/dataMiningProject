#Online shoppers Intention detection

df<-online_shoppers_intention

is.factor(df$VisitorType)
is.factor(df$Weekend)
is.factor(df$Revenue)

as.factor(df$VisitorType)
as.factor(df$Weekend)
as.factor(df$Revenue)

as.numeric(df$VisitorType)
levels(df$VisitorType)
as.numeric(df$Weekend)
levels(df$Weekend)
as.numeric(df$Revenue)

mydf<-as.data.frame(lapply(df, as.numeric))

#finding the correlation matrix

correlation<-cor(mydf)
round(correlation, 2)
install.packages("corrplot")
library(corrplot)
corrplot(correlation, method = "number")
corrplot(correlation, type="upper", order="hclust", tl.col="black", tl.srt=45)


#SPLITTING THE DATASET INTO TRAINING AND TEST SET
install.packages('caTools')
library('caTools')
set.seed(123)
split = sample.split(mydf$Revenue, SplitRatio = 0.75)
trainingset = subset(mydf, split == TRUE)
testset = subset(mydf, split == FALSE)


##########FITTING LOGISTIC REGRESSION


logistic_classifier = glm(formula = Revenue ~ .,
                          family = binomial,
                          data = trainingset)
#PREDICTING THE TEST_SET RESULTS
prob_predict = predict(logistic_classifier, type = 'response', newdata = testset[-18])
prob_predict #probabilities of each of the testset observation
y_pred = ifelse(prob_predict > 0.5, 1, 0) #Converting the probabilities interms of 0 and 1
y_pred
y_actual = testset[,18]
as.numeric(y_actual)
as.numeric(y_pred)
logistic_confusion = confusionMatrix(cm, positive = '1')
logistic_confusion

#CONFUSION MATRIX
cm  = table(testset[,18], y_pred)
cm

#ROC CURVE OF LOGISTIC FUNCTION
library(pROC)
roc_graph<-roc(y_actual, y_pred)
roc_graph
plot.roc(roc_graph, print.auc = TRUE)


install.packages('tidyverse')
library(tidyverse)
install.packages('caret')
library(caret)
levels(y_pred)
logistic_confusion = confusionMatrix(cm, positive = '1')

#MODEL ACCURACY
accuracy = sum(diag(cm))/sum(cm)
accuracy


############K-NEAREST NEIGHBOURS ALGORITHMS



#IMPLEMENTING K-NEAREST NEIGHBOURS
install.packages('caTools')
library('caTools')
library(class)
y_pred_knn = knn(train = trainingset[, -18], test = testset[, -18],
                 cl = trainingset[, 18], k = 96)

y_pred_knn
#CONFUSION MATRIX FOR K-NN
cm_knn  = table(testset[,18], y_pred_knn)
cm_knn
knn_confusion = confusionMatrix(cm_knn, positive = '1')
knn_confusion


# MODEL ACCURACY FOR K-KNN
accuracy_knn = sum(diag(cm_knn))/sum(cm_knn)
accuracy_knn

############IMPLEMENTING SUPPORT VECTOR MACHINE

install.packages('e1071')
library('e1071')
svm_classifier = svm(formula = Revenue ~ .,
                     data = trainingset, 
                     type = 'C-classification',
                     kernel = 'radial')
svm_y_pred = predict(svm_classifier, newdata = testset[, -18])
svm_y_pred
#SVM CONFUSION MATRIX
cm_SVM = table(testset[, 18], svm_y_pred)
cm_SVM
SVM_confusion = confusionMatrix(cm_SVM, positive = '1')
SVM_confusion
#SVM MODEL ACCURACY
accuracy_SVM = sum(diag(cm_SVM))/sum(cm_SVM)
accuracy_SVM

##############DECISION TREE CLASSIFICATION MODEL

install.packages('rpart')
library('rpart')
decisiontree_classifier = rpart(formula = Revenue ~ .,
                                data = trainingset)
decisiontree_y_pred = predict(decisiontree_classifier, newdata = testset[, -18])
decisiontree_y_pred
decisiontree_final_y = ifelse(decisiontree_y_pred > 0.5, 1, 0) #Converting the probabilities interms of 0 and 1
decisiontree_final_y

#CONFUSION MATRIX FOR DECISION TREE
cm_decision_tree = table(testset[, 18], decisiontree_final_y)
cm_decision_tree
decision_confusion = confusionMatrix(cm_decision_tree, positive = '1')
decision_confusion

#DECISION TREE MODEL ACCURACY
accuracy_decision_tree = sum(diag(cm_decision_tree))/sum(cm_decision_tree)
accuracy_decision_tree

############CLUSTERING
datascaled<-scale(mydf[])
pca = preProcess(x= datascaled[,-18], method = 'pca',pcaComp = 2)
datascaled = predict(pca, datascaled)

col_order <- c("PC1", "PC2", "Revenue")
datascaled <- datascaled[, col_order]

#K-Means clustering
fitk<-kmeans(datascaled,2)
fitk

k<-list()
for(i in 1:20){
  k[[i]]<-kmeans(datascaled, i)
}
k

betweenss_totss<-list()
for(i in 1:20){
  betweenss_totss[[i]]<-k[[i]]$betweenss/k[[i]]$totss
}
betweenss_totss
plot(1:20,betweenss_totss,type="b",ylab = "between ss/total ss", xlab = "clusters(k)")
for(i in 1:4){
  plot(datascaled,col = k[[i]]$cluster)
}

#hierarchial clustering
dissimilarity_structure<-dist(datascaled)
fit_hc<-hclust(dissimilarity_structure, "ward.D2")
fit_hc
plot(fit_hc) #Dendograms


#density based clustering
install.packages("dbscan")
library("dbscan")
kNNdistplot(datascaled, k=2)
abline(h = 1,col = 'red',lty = 2)
fitD<-dbscan(datascaled, eps = 1,minPts = 18)
fitD
plot(datascaled, col = fitD$cluster)

