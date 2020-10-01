#Classification and Regression Trees
library(mlr3verse)
library(mlbench)
library(ggplot2)
library(rpart.plot)

set.seed(334)
spirals <- as.data.frame(mlbench.spirals (n = 500, sd = 0.1))
spirals_task <- TaskClassif$new(id = "spirals", backend = spirals, target = "classes")

# Visualization of the data
ggplot(data = spirals, aes(x.1, x.2, color = classes)) + geom_point()

#Learner
learner_cart <- lrn("classif.rpart", minsplit = 20, cp = 0.01)
learner_cart$train(spirals_task)

#Visualize fitted tree
rpart.plot(learner_cart$state$model, roundint = FALSE)
plot_learner_prediction(learner = learner_cart, task = spirals_task)

# Choose different observations for training the model
n = nrow(spirals)
train_idx <- sample(seq_len(n), 0.8*n)
test_idx <- setdiff(seq_len(n), train_idx)
spirals_task <- TaskClassif$new(id = "spirals_task", backend = spirals, target = "classes")

learner_cart <- lrn("classif.rpart", minsplit = 20, cp = 0.01)
learner_cart$train(spirals_task, row_ids = train_idx)

rpart.plot(learner_cart$state$model, roundint = FALSE)
plot_learner_prediction(learner = learner_cart, task = spirals_task)

# Choose different hyperparameter configurations
minsplit <- 5
cp <- 0.001

learner_cart <- lrn("classif.rpart", minsplit = minsplit, cp = cp)
learner_cart$train(spirals_task)

rpart.plot(learner_cart$state$model, roundint = FALSE)
plot_learner_prediction(learner = learner_cart, task = spirals_task)