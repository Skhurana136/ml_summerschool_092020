#install.packages ("ggplot2")
#install.packages ("mlr3verse")
#install.packages ("mlr3learners")
#install.packages("kknn")
library(ggplot2)
library(mlr3)
library(mlr3learners)
library(kknn)
library(mlr3viz)

cols <- list("sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings")

data <- read.csv("Y:/Home/khurana/1.Scripts/ml_summerschool_092020/datasets/abalone.data", header = FALSE, col.names = cols)
head(data)

#a) Plot LongestShell, WholeWeight on the x- and y-axis and color points with Rings
ggplot() + geom_point(data = data, aes(x = length, y = whole_weight, color = rings))

#Using the mlr3-package:
#b) Fit a linear model
lm_task <- TaskRegr$new(id = "lm_task", 
                        backend = data[,c("rings", "length", "whole_weight")], 
                        target = "rings")

lm_learner <- lrn("regr.lm")
lm_learner$train(lm_task)
pred_lm <- lm_learner$predict(lm_task)

#c) Fit a k-nearest-neighbors model
kn_task <- TaskRegr$new(id = "kn_task", 
                        backend = data[,c("rings", "length", "whole_weight")], 
                        target = "rings")

kknn_learner <- lrn("regr.kknn", k = 3)

kknn_learner$train(kn_task)
pred_knn <- kknn_learner$predict(kn_task)


#d) Plot the prediction surface of lm and of knn (Hint: Use autoplot())
autoplot(pred_lm)
autoplot(pred_knn)