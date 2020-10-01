library(mlr3)
library(mlr3learners)

#dataset:
df_banana <- data.frame(
  Color = c("yellow", "yellow", "yellow", "brown", "brown", "green", "green", "red"),
  Form = c("oblong", "round", "oblong", "oblong", "round", "round", "oblong", "round"),
  Origin = c("imported", "domestic", "imported", "imported", "domestic", "imported",
             "domestic", "imported"),
  Banana = c("yes", "no", "no", "yes", "no", "yes", "no", "no")
)

#Task: to classify a new data point/sample

#Load learner:
nb_learner <- lrn("classif.naive_bayes",
                  predict_type = "prob")
#Define training task:
banana_task <- TaskClassif$new(
  id = "banana",
  backend = df_banana,
  target = "Banana"
)

#Train learner and get a model:
nb_learner$train(banana_task)

#Define prediction task:
#Setup features:
ufo <- data.frame(Color = "yellow", Form = "round", Origin = "imported", Banana = NA)
df_banana <- rbind(df_banana, ufo)

#Set up new task to give to the old model:
ufo_task <- TaskClassif$new(
  id = "ufo_task",
  backend = df_banana,
  target = "Banana"
)

#Predict:
nb_learner$predict(ufo_task)