#Exercise_4_1
library(mlr3verse)
library(rpart.plot)

#) Take a look at the spam dataset (?mlr3::mlr tasks spam). Shortly describe what kind of classification
#problem this is and access the corresponding task predefined in mlr3.
?mlr3::mlr_tasks_spam
#Task type - binary classification
#classifies emails as spam or non-spame. 57 variables

spamtask <- tsk("spam")

#b) Use a decision tree to predict spam. Try refitting with different samples. How stable are the trees?
#  Hint: Use rpart.plot() from the package rpart.plot to vizualize the trees.
#(You can access the model of a learner by its class attribute model)
cart_learner <- lrn("classif.rpart")
cart_learner$train(spamtask)
rpart.plot(cart_learner$model)
#c) Use the random forest learner classif.ranger to fit the model and state the oob-error.
rf_learner <- lrn("classif.ranger", "oob.error" = TRUE)
rf_learner$train(spamtask)
rf_learner$predict(spamtask)
rf_learner$oob_error()
model <- rf_learner$model
model$prediction_error

#d) Your boss wants to know which variables have the biggest influence on the prediction quality.
#Explain your approach in words as well as code.
#Hint: use an adequate variable importance filter as described in https://mlr3filters.mlr-org.com/
#variable-importance-filters
learner <- lrn("classif.ranger", importance = "permutation", "oob.error" = TRUE)
filter <- flt("importance", learner = learner)
filter$calculate(tsk("spam"))
head(as.data.table(filter))


                                                                              
                                                                              