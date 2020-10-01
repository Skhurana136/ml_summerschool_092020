#Evaluation_Classification
library(mlr3verse)

tsk("pima")$head()

?mlbench::PimaIndiansDiabetes

pima_task <-  tsk("pima")
#filter to get complete cases only
pima_task$filter(rows = which(complete.cases(pima_task$data())))
#logistic regression on the whole task:
learner <- lrn("classif.log_reg", predict_type = "prob")
learner$train(task = pima_task)
#Predict model
model_prediction <-learner$predict(task = pima_task)

#Confusion matrix, using class method set_threshold to vary decision threshold used
model_prediction$confusion

#change threshold
for x in (0,0.1,0.3,0.5, 0.7) (
  model_prediction$set_threshold(x)
  model_prediction$confusion
  autoplot(model_prediction, "roc")
  # To calculate the performance on a prediction object use its class method 'score()' 
  model_prediction$score(list(msr("classif.auc"), msr("classif.ce")))
)

class <- sample(c(1, 2), size = 100, replace = TRUE)
df_sim <- data.frame(x = rnorm(100, mean = 2*class, sd = class), y = rnorm(100, mean = 2*class, sd = class), class = as.character(class))

task_sim <- TaskClassif$new(id = "2_gaussians", backend = df_sim, target = "class")
kknn_learner <- lrn("classif.kknn", k = 1)
plot_learner_prediction(learner = kknn_learner, task = task_sim)