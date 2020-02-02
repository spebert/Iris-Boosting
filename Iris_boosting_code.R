
# Final Project Stat 666

# Load libraries
library(tidyverse)
library(xgboost)
library(randomForest)
library(mvtnorm)
library(xtable)

# Multiple classification on Iris data using XGBoost
# Iris data is already loaded into R
# I will use one vs all classification for the project

# Prep the data
irisdat <- iris
irisdat$setosa <- as.numeric(irisdat$Species=="setosa")
irisdat$versicolor <- as.numeric(irisdat$Species=="versicolor")
irisdat$virginica <- as.numeric(irisdat$Species=="virginica")
irisdat$SpeciesInt <- as.integer(irisdat$Species) - 1

# Split into test and training datasets
set.seed(12)
train.rows <- sample(1:nrow(irisdat),100,replace=FALSE)
iris.train <- irisdat[train.rows,]
iris.test <- irisdat[-train.rows,]


# Xgboost multiclass classification
xgb.multiclass <- xgboost(data=data.matrix(iris.train[,1:4]),
                          label=data.matrix(iris.train$SpeciesInt),
                          eta=0.01,
                          max_depth=5,
                          nround=100,
                          objective="multi:softprob",
                          num_class=3)
multiclass.xgb.pred <- predict(xgb.multiclass,data.matrix(iris.test[,1:4]),reshape=TRUE)
assigned.iris.multiclass <- apply(multiclass.xgb.pred,1,function(x) which.max(x))
# Prediction performance
mean(recode(assigned.iris.multiclass,'1'="setosa",'2'="versicolor",'3'="virginica") == iris.test$Species)

# Now look how random forests compare
rf.iris <- randomForest(data=iris.train,Species~Sepal.Length+Sepal.Width+Petal.Length+Petal.Width)
pred.rf <- predict(rf.iris,data.matrix(iris.test[,1:4]))
mean(pred.rf==iris.test$Species)


# Use discriminant function for classification now
# Assuming equal costs
group_counts <- iris.train %>%
  group_by(SpeciesInt) %>%
  summarise(n=n())
# compute s pooled
# calculate covariance, and n for each group
S0_pre <- iris.train %>%
  filter(SpeciesInt==0)
S0 <- cov(S0_pre[,1:4,])
S1_pre <- iris.train %>%
  filter(SpeciesInt==1)
S1 <- cov(S1_pre[,1:4,])
S2_pre <- iris.train %>%
  filter(SpeciesInt==2)
S2 <- cov(S2_pre[,1:4,])
n0 <- drop(data.matrix(group_counts[1,2]))
n1 <- drop(data.matrix(group_counts[2,2]))
n2 <- drop(data.matrix(group_counts[3,2]))
# Pool the covariance matrices
Spooled <- ( n0*S0 + n1*S1 + n2*S2 ) / (n0+n1+n2-3)
SpooledInv <- solve(Spooled)
# Linear classification function
xbar0 <- apply(subset(iris.train,SpeciesInt==0)[,1:4],2,mean)
xbar1 <- apply(subset(iris.train,SpeciesInt==1)[,1:4],2,mean)
xbar2 <- apply(subset(iris.train,SpeciesInt==2)[,1:4],2,mean)
linclassfuncs <- cbind(t(t(xbar0)%*%SpooledInv %*% t(iris.test[,1:4])) - 0.5 * (t(xbar0)%*%SpooledInv%*%t(t(xbar0)))[1,1],
                       t(t(xbar1)%*%SpooledInv %*% t(iris.test[,1:4])) - 0.5 * (t(xbar1)%*%SpooledInv%*%t(t(xbar1)))[1,1],
                       t(t(xbar2)%*%SpooledInv %*% t(iris.test[,1:4])) - 0.5 * (t(xbar2)%*%SpooledInv%*%t(t(xbar2)))[1,1])

# Check prediction performance
assigned.lin.func <- apply(linclassfuncs,1,function(x) which.max(x))
mean(recode(assigned.lin.func,'1'="setosa",'2'="versicolor",'3'="virginica") == iris.test$Species)


# All 3 Methods performed about the same. xgb:96% rf:96% lin:98%
# This is before I run the models multiple times

# Cross Validation for the iris data
Ndraws <- 1000

# Matrix to store the predicted probabilities
iris.all.test <- matrix(nrow=Ndraws,numeric(3*Ndraws))

set.seed(12)
for (i in 1:Ndraws) {
  iris.all.test[i,] <- all_methods(irisdat$SpeciesInt,irisdat[,1:4])
  print(i)
}

# Results
apply(iris.all.test,2,mean)


# Simulation study to test the effectiveness of these methods
# Parameters to adjust
#   number of predictors: xn
#   noise in the data: sigma2
#   correlation: rho
#   number of data points: n (for each class)

# Create datasets function
makeDat <- function(xn,sigma2,rho,n) {
  corrMat <- matrix(nrow=xn,rep(rho,xn^2))
  diag(corrMat) <- rep(1,xn)
  covMat <- sigma2*corrMat
  mean0 <- rep(0,xn)
  mean1 <- rep(1,xn)
  mean2 <- rep(2,xn)
  the.predictors <- rbind(rmvnorm(n,mean0,covMat),
                          rmvnorm(n,mean1,covMat),
                          rmvnorm(n,mean2,covMat))
  the.response <- matrix(ncol=1,c(rep(0,n),rep(1,n),rep(2,n)))
  list(the.response,the.predictors)
}

# Function for doing all three of the tests from a set of data
#   Make sure that the response is in integers 0,1, and 2 and match the rows for the predictor rows
all_methods <- function(response,predictors) {
  train.rows <- sample(1:nrow(predictors),nrow(predictors)*2/3,replace=FALSE)
  train.predictors <- predictors[train.rows,]
  train.response <- response[train.rows]
  test.predictors <- predictors[-train.rows,]
  test.response <- response[-train.rows]
  
  # xgb prediction
  xgb.fit <- xgboost(data=data.matrix(train.predictors),
                     label=data.matrix(train.response),
                     eta=0.01,
                     max_depth=5,
                     nround=100,
                     objective="multi:softprob",
                     num_class=3,
                     verbose=0)
  xgb.pred <- predict(xgb.fit,data.matrix(test.predictors),reshape=TRUE)
  assigned.class.xgb <- apply(xgb.pred,1,function(x) which.max(x)) - 1
  
  # randomForest prediction
  rf.fit <- randomForest(x=data.matrix(train.predictors),
                         y=as.factor(data.matrix(train.response)),
                         ntree=500)
  rf.pred <- predict(rf.fit,data.matrix(test.predictors))
  
  # linear discriminant function
  rows_for0 <- which(train.response==0)
  rows_for1 <- which(train.response==1)
  rows_for2 <- which(train.response==2)
  n0 <- length(rows_for0)
  n1 <- length(rows_for1)
  n2 <- length(rows_for2)
  xbar0 <- apply(train.predictors[rows_for0,],2,mean)
  xbar1 <- apply(train.predictors[rows_for1,],2,mean)
  xbar2 <- apply(train.predictors[rows_for2,],2,mean)
  S0 <- cov(train.predictors[rows_for0,])
  S1 <- cov(train.predictors[rows_for1,])
  S2 <- cov(train.predictors[rows_for2,])
  Spooled <- ( n0*S0 + n1*S1 + n2*S2) / (n0+n1+n2-3)
  SpooledInv <- solve(Spooled)
  linclassfuncs <- cbind(t(t(xbar0)%*%SpooledInv %*% t(test.predictors)) - 0.5 * (t(xbar0)%*%SpooledInv%*%t(t(xbar0)))[1,1],
                         t(t(xbar1)%*%SpooledInv %*% t(test.predictors)) - 0.5 * (t(xbar1)%*%SpooledInv%*%t(t(xbar1)))[1,1],
                         t(t(xbar2)%*%SpooledInv %*% t(test.predictors)) - 0.5 * (t(xbar2)%*%SpooledInv%*%t(t(xbar2)))[1,1])
  lin.pred <- apply(linclassfuncs,1,function(x) which.max(x))-1
  
  # Prediction performance for each method
  xgb.correct <- mean(assigned.class.xgb==test.response)
  rf.correct <- mean(rf.pred==test.response)
  lin.correct <- mean(lin.pred==test.response)
  
  c(xgb.correct,rf.correct,lin.correct)
  
}

# Checking function to see if I get same results as above
set.seed(12)
all_methods(irisdat$SpeciesInt,irisdat[,1:4])
# All good :)

# Checking combination of functions
datMad <- makeDat(4,20,0.5,100)
all_methods(datMad[[1]],datMad[[2]])

# Parameters I want to check
# xn predictors: 2,4,8
# sigma2: 0.01,0.05,0.1,0.5,1,5,10,20
# rho: 0.01,0.5,0.99
# n: 50,100
xn.check <- c(2,4,8)
sigma2.check <- c(0.01,0.05,0.1,0.5,1,5,10,20)
rho.check <- c(0.01,0.5,0.99)
n.check <- c(50,100)
all.checks <- expand.grid(xn.check,sigma2.check,rho.check,n.check)

# Simulation
# Initialize parameters
# Number of times to run for each dataset
Ndraws <- 100
# Success rate for each of the three methods stored in this matrix
success.rates <- matrix(nrow=nrow(all.checks),numeric(nrow(all.checks)*3))

set.seed(2)
for (i in 1:nrow(all.checks)) {
  # Matrix for success rate after each iteration
  ind.success.rates <- matrix(nrow=Ndraws,numeric(Ndraws*3))
  for (j in 1:Ndraws) {
    sim.dat <- makeDat(all.checks[i,1],all.checks[i,2],all.checks[i,3],all.checks[i,4])
    ind.success.rates[j,] <- all_methods(sim.dat[[1]],sim.dat[[2]])
    if (j%%25==0) print(j)
  }
  success.rates[i,] <- apply(ind.success.rates,2,mean)
  print(i)
}

# All of the data created was pretty well behaved so now I look at the data but they don't have the 
#   same covariance matrices
# Function for making messy data
makeDatMess <- function(xn,sigma2,rho,n) {
  corrMat0 <- matrix(nrow=xn,rep(rho,xn^2))
  diag(corrMat0) <- rep(1,xn)
  corrMat1 <- matrix(nrow=xn,rep(-rho,xn^2))
  diag(corrMat1) <- rep(1,xn)
  dif.sigmas <- sigma2*c(0.5,1,3)
  covMat0 <- dif.sigmas[1]*corrMat0
  covMat1 <- dif.sigmas[2]*corrMat1
  covMat2 <- dif.sigmas[3]*corrMat0
  mean0 <- rep(0,xn)
  mean1 <- rep(1,xn)
  mean2 <- rep(2,xn)
  the.predictors <- rbind(rmvnorm(n,mean0,covMat0),
                          rmvnorm(n,mean1,covMat1),
                          rmvnorm(n,mean2,covMat2))
  the.response <- matrix(ncol=1,c(rep(0,n),rep(1,n),rep(2,n)))
  list(the.response,the.predictors)
}

# Success rate for each of the three methods stored in this matrix for different matrices
success.rates.messy <- matrix(nrow=nrow(all.checks),numeric(nrow(all.checks)*3))

set.seed(2)
for (i in 1:nrow(all.checks)) {
  # Matrix for success rate after each iteration
  ind.success.rates <- matrix(nrow=Ndraws,numeric(Ndraws*3))
  for (j in 1:Ndraws) {
    sim.dat <- makeDatMess(all.checks[i,1],all.checks[i,2],all.checks[i,3],all.checks[i,4])
    ind.success.rates[j,] <- all_methods(sim.dat[[1]],sim.dat[[2]])
    if (j%%25==0) print(j)
  }
  success.rates.messy[i,] <- apply(ind.success.rates,2,mean)
  print(i)
}


# Save the data so I don't have to run it again :)
save(success.rates,success.rates.messy,
     file=file.path('C:/Users/spenc/Documents/Fall2019/Stat 666 Multivariate Statistical Methods/FinalProj666/TheProj/clean_dat_success_rates.Rdata'))
load('C:/Users/spenc/Documents/Fall2019/Stat 666 Multivariate Statistical Methods/FinalProj666/TheProj/clean_dat_success_rates.Rdata')


# Results from the simulation
the_dat <- data.frame(cbind(all.checks,success.rates))
colnames(the_dat) <- c("xn","s2","rho","n","boost","rf","lc")
subset(the_dat,xn==4 & rho==0.5 & n==50)

# Find values I want to use in the results
subset(the_dat,xn==4 & rho==0.5 & n==100 & s2==0.5)
subset(the_dat,xn==8 & rho==0.5 & n==100 & s2==0.5)
subset(the_dat,xn==8 & rho==0.5 & n==100 & s2==0.5)
subset(the_dat_messy,xn==8 & rho==0.5 & n==100 & s2==5)

# Table for report
the_dat_messy <- data.frame(cbind(all.checks,success.rates.messy))
colnames(the_dat_messy) <- c("xn","s2","rho","n","boost","rf","lc")

xtable(cbind(the_dat[c(29,35,44,106,107,108,111,114),],the_dat_messy[c(29,35,44,106,107,108,111,114),5:7]))
