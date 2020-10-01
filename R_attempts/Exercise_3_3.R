library(mlr3verse)

bh_task <- tsk("boston_housing")

n <- bh_task$nrow
# select index vectors to subset the data randomly
learner <- lrn("regr.kknn", k = 3)

x <- 321
set.seed(321)
train_ind <- sample(seq_len(n), 0.8*n)
test_ind <- setdiff(seq_len(n), train_ind)
# specify learner
# train model to the training set
learner$train(bh_task, row_ids = train_ind)
# predict on the test set
pred <- learner$predict(bh_task, row_ids = test_ind)
pred

pred_train <- learner$predict(bh_task, row_ids = train_ind)
pred_train$score(list(msr("regr.mse"), msr("regr.mae")))
autoplot(pred_train)
plot_learner_prediction(learner = learner, task = bh_task, )
pred_test <- learner$predict(bh_task, row_ids = test_ind)
pred_test$score(list(msr("regr.mse"), msr("regr.mae")))
autoplot(pred_test)
rdesc <- rsmp("cv", folds = 10)
r <- resample(bh_task, learner, rdesc)

r$aggregate(list(msr("regr.mse"), msr("regr.mae")))