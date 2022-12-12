
library("ggplot2")
library("e1071")
library(dplyr)
library(reshape2)
library(corrplot)
library(caret)
library(pROC)
library(gridExtra)
library(magrittr) 
library(ROCR)
library(grid)
library(ggfortify)
library(purrr)
library(nnet)
library(doParallel) # parallel processing 
registerDoParallel()
require(foreach)
require(iterators) 
require(parallel)


#Load Data set

breastcancer_rawdata <- read.csv(file.choose(), sep = ",")


# Viewing all observations of the raw dataset
View(breastcancer_rawdata)

View(head(breastcancer_rawdata))
glimpse(breastcancer_rawdata)

# statistical insight and structure of raw dataset
str(breastcancer_rawdata)

# Summary of Raw dataset
summary(breastcancer_rawdata)

# Dimension of raw dataset - 32 Variables

dim(breastcancer_rawdata) 
sapply(breastcancer_rawdata, function(missingv) sum(is.na(missingv)))

# 33 Variables in the summary while shows 33 variables. 
# Get rid of id as it's redundant 
# Get rid of NA's
# Missing Values 

# Removing ID's and NA's
breastcancer_data <- breastcancer_rawdata[-c(1, 33)]


dim(breastcancer_data)
glimpse(breastcancer_data)
# We now have 31 Variables

#Converting data to factors and tidying the data

breastcancer_data$diagnosis <- as.factor(breastcancer_data$diagnosis)

summary(breastcancer_data)
head(breastcancer_data)

#checking for missing variables
sapply(breastcancer_data, function(missingv) sum(is.na(missingv)))



# No missing Variables 

# 357 Benign and 212 Malignant 
# 31 Columns, 569 records 

# Data Visulisation 

# Cancer diagnosis frequency 

DiagnosisTable <- table(breastcancer_data$diagnosis)
colors <- terrain.colors(3)

# Pie Chart 
DiagnosisPropTable <- prop.table(DiagnosisTable)*100
DiagnosisPropDF  <- as.data.frame(DiagnosisPropTable)
chartlabels <- sprintf("%s - %3.1f%s", DiagnosisPropDF[,1],DiagnosisPropTable,"%" )
pie(DiagnosisPropTable, labels = chartlabels, clockwise = TRUE, col = colors, border = "gainsboro", radius = 0.8, cex = 0.8, main = "Cancer Diagnosis Frequency Pie Chart"   )
legend(1, .4, legend = DiagnosisPropDF[,1],cex = 0.7, fill = colors)





#distribution of the numeric variables
breastcancer_data_num=subset(breastcancer_data,select = -diagnosis)

#Break up columns into groups, according to their suffix designation
#(_mean, _se,and __worst) to perform visualisation plots off.

meandata <- breastcancer_data[ ,c("diagnosis", "radius_mean", "texture_mean","perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave.points_mean", "symmetry_mean", "fractal_dimension_mean")]
sedata <- breastcancer_data[ ,c("diagnosis", "radius_se", "texture_se","perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave.points_se", "symmetry_se", "fractal_dimension_se" )]
worstdata <- breastcancer_data[ ,c("diagnosis", "radius_worst", "texture_worst","perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave.points_worst", "symmetry_worst", "fractal_dimension_worst" )]

#Plot histograms of "_mean" variables group by diagnosis
ggplot(data = melt(meandata, id.var = "diagnosis"), mapping = aes(x = value)) + geom_histogram(bins = 10, aes(fill=diagnosis), alpha=0.5) + facet_wrap(~variable, scales ='free_x') +  ggtitle("_mean Histogram")

#Plot histograms of "_se" variables group by diagnosis
ggplot(data = melt(sedata, id.var = "diagnosis"), mapping = aes(x = value)) + geom_histogram(bins = 10, aes(fill=diagnosis), alpha=0.5) + facet_wrap(~variable, scales = 'free_x') +  ggtitle("_se Histogram")


#Plot histograms of "_worst" variables group by diagnosis
ggplot(data = melt(worstdata, id.var = "diagnosis"), mapping = aes(x = value)) + geom_histogram(bins = 10, aes(fill=diagnosis), alpha=0.5) + facet_wrap(~variable, scales = 'free_x') + ggtitle("_worst Histogram")


#inference-most of the variables are skewed


#Corelation Plot
#Collinearity Calculation
Cor <- cor(breastcancer_data[,2:31])
corrplot(Cor, order = "hclust", tl.cex = 0.9)

# Plots the features of the 31 variables
# Shows correlation between variables that are highly correlated which may show data redundancy 
# for example texture_mean and texture_worse

# Highly correlated can be 0.7 +
highlycorrelated <- colnames(breastcancer_data)[findCorrelation(Cor, cutoff = 0.9, verbose = TRUE)] #0.9

breastcancer_data_cor <- breastcancer_data[, which(!colnames(breastcancer_data) %in% highlycorrelated)] 
ncol(breastcancer_data_cor)

# Pre Process Data 

breastcancer_data_cor <- cbind(diagnosis = breastcancer_data$diagnosis, breastcancer_data_cor)
head(df)

breastcancer_data_PP <- preProcess(x = breastcancer_data_cor,method = c("center", "scale")) 

ProcessedcancerData <- predict(breastcancer_data_PP, breastcancer_data_cor) #stored
head(ProcessedcancerData)

set.seed(1) #for reproductibility of results
# y.factor = target variable p = amount of training data used
trainIndex <- caret::createDataPartition( ProcessedcancerData$diagnosis,times = 1,p = 0.75,list = FALSE) #.factor

cancerTrain <- ProcessedcancerData[trainIndex,]
cancerTest <- ProcessedcancerData[-trainIndex,]
testValues <- cancerTest$diagnosis #.factor


set.seed(1)
cancerFitAll <- glm(diagnosis ~.,family = binomial(link = "logit"),data = breastcancer_data_cor) #.factor
summary(cancerFitAll)

# the importance of each variable in model 
varImp(cancerFitAll) 

# look into dropping perimeter_mean, radius_se, concave.points_worst as they have the lowest importance
set.seed(1)
cancerFitFew <- glm(diagnosis ~. -perimeter_mean -radius_se -concave.points_worst,family = binomial(link = "logit"),data = breastcancer_data_cor)
summary(cancerFitFew)

# Analysis of Variance(ANOVA)

anova(cancerFitAll, cancerFitFew, test="Chisq")

# Deviance show that they are basically the same 

set.seed(1)
#number of folds is 10 by default
FC <- trainControl(method = "repeatedcv",
                           
                           repeats = 3, 
                           savePredictions = T)

glmCancerFit <- train(diagnosis.formuk ~.-perimeter_mean -radius_se -concave.points_worst, 
                      data = cancerTrain,
                      method = "glm",
                      family = "binomial",
                      trControl = FC) # -perimeter_mean -radius_se -concave.points_worst

glmFitAcc <- train(diagnosis ~.-perimeter_mean -radius_se -concave.points_worst, 
                   data = cancerTrain,
                   method = "glm",
                   metric = "Accuracy",
                   trControl = FC) %>% print

# Linear Regression Accuracy = 96.6%/97%

# Work Out Linear Regression AUC

ROCprobability <- predict(cancerFitFew, newdata = cancerTest,type="response")

ROCprediction <- prediction(ROCprobability,testValues)

ROCperformance <- performance(ROCprediction, measure = 'tpr',x.measure = 'fpr')

# ROC Graph
plot(ROCperformance, lwd = 2, colorize = TRUE, main = "GLM ROC Curve") # rainbow!
lines(x = c(0, 1), y = c(0, 1), col = "grey78", lwd = 1) 

# Set up area under curve variable
auc = performance(ROCprediction, measure = "auc") %>% print
# Print area under curve 
auc = auc@y.values[[1]] %>% print

# Sensitivity/Specificity Curve


plot(ROCperformance,avg= "threshold",colorize=TRUE,lwd= 3, main=" Sensitivity/Specificity plots")
plot(ROCperformance,lty=3,col="grey78",add=TRUE)

sens = performance(ROCprediction, measure = "sens") 
# Print area under curve
sens = sens@y.values[[1]] %>% print


# LR Area under ROC curve = 0.8054908 - 81%

#Random forest: caret

# FCProb: used  for AUROC of RF later
set.seed(1)
FCProb <- trainControl(method = "repeatedcv",
                               repeats = 3, 
                               savePredictions = T, 
                               classProbs = T, # probability instead of response
                               summaryFunction = twoClassSummary)

rfCancerFit <- train(diagnosis ~.-perimeter_mean -radius_se -concave.points_worst, 
                     data = cancerTrain,
                     method = "rf",
                     trControl = FC)

rfFitAcc <- train(diagnosis ~.-perimeter_mean -radius_se -concave.points_worst, 
                   data = cancerTrain,
                   method = "rf",
                   metric = "Accuracy",
                   trControl = FC) %>% print
# 96% accuracy for Random Forest 96.6%/97%



rfFitROC <- train(diagnosis ~.--perimeter_mean -radius_se -concave.points_worst, 
                  data = cancerTrain,
                  method = "rf",
                  metric = "ROC",
                  trControl = FCProb) %>% print

# 99%  AUROC for RF

# Sensitivity 0.9736942   -  97% for RF

#Specificity     0.9266667 - 93% for RF

# 96% accuracy for Random Forest 

# 81% AUROC for GLM - Linear Regression 

# Sensitivity for Linear Regression 

# Specificity for Linear Regression 

# 96.6%/97% Accuracy for Linear Regression

# The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes

# Therefore despite the fact that LR has higher accuracy, the AUC proves that RF is more reliable

