library(mlr3)
library(mlr3learners)
library(mlbench)
data(Satellite)
satellite_task <-
  TaskClassif$new(id = "satellite_task",
                  backend = Satellite,
                  target = "classes")

knn_learner <- lrn("classif.kknn", k = 3)

# Train and test subsets:
set.seed(42)
train_indices <-
  sample.int(nrow(Satellite), size = 0.8 * nrow(Satellite))
train_indices <-
  sample.int(nrow(Satellite), size = 0.8 * nrow(Satellite))
test_indices <- setdiff(1:nrow(Satellite), train_indices)

# Training data performance estimate
knn_learner$train(task = satellite_task, row_ids = train_indices)
pred_train <- knn_learner$predict(task = satellite_task, row_ids = train_indices)
train = pred_train$score()

# Test data performance estimate
pred_test <- knn_learner$predict(task = satellite_task, row_ids = test_indices)
test = pred_test$score()

#Cross-validaton
rdesc <- rsmp("cv", folds = 10)
res <- resample(satellite_task, knn_learner, rdesc)
res$score()
cv = res$aggregate()

print(train)
print(test)
print(cv)