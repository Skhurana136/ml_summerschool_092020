#EXERCISE 4_2
#Classification and Regression Trees
library(mlr3verse)
library(mlbench)
library(ggplot2)
library(rpart.plot)
library (gridExtra)

set.seed(334)
spirals <- as.data.frame(mlbench.spirals (n = 500, sd = 0.1))
spirals_task <- TaskClassif$new(id = "spirals", backend = spirals, target = "classes")

# Visualization of the data
ggplot(data = spirals, aes(x.1, x.2, color = classes)) + geom_point()

#Vary learners and plot

gg1 <- plot_learner_prediction(learner = lrn("classif.ranger", num.trees = 1, predict_type = "prob"),
                               task = spirals_task) + guides(alpha = "none", shape = "none")
gg10 <- plot_learner_prediction(learner = lrn("classif.ranger", num.trees = 5, predict_type = "prob"),
                               task = spirals_task) + guides(alpha = "none", shape = "none")
gg100 <- plot_learner_prediction(learner = lrn("classif.ranger", num.trees = 10, predict_type = "prob"),
                                task = spirals_task) + guides(alpha = "none", shape = "none")+ ggtitle("10")
gg500 <- plot_learner_prediction(learner = lrn("classif.ranger", num.trees = 100, predict_type = "prob"),
                                task = spirals_task) + guides(alpha = "none", shape = "none") + ggtitle("100")
gg1000 <- plot_learner_prediction(learner = lrn("classif.ranger", num.trees = 1000, predict_type = "prob"),
                                task = spirals_task) + guides(alpha = "none", shape = "none")

gridExtra::grid.arrange(gg1, gg10, gg100, gg500, ncol = 2, nrow = 2)
rf_learner <- lrn("classif.ranger", num.trees = 100, predict_type = "prob")
rf_learner$train(spirals_task)
rf_learner$predict(spirals_task)


treelist <- list(1,10,50,100,500,1000)
for (t in treelist) {
  rf_learner_t <- lrn(id = t, "classif.ranger", num.trees = t, predict_type = "prob")
  rf_learner_t$train(spirals_task)
  pred <- rf_learner_t$predict(spirals_task)
#  print(rf_learner_t$oob_error())
  print(pred$score(list(msr("classif.bbrier"),msr("classif.ce"))))
}