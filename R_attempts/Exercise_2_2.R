#Choose some of the classifiers already introduced in the lecture 
#and visualize their decision boundaries for relevant hyperparameters.
#Use mlbench::mlbench.spirals to generate data and use plot learner prediction for visualization.
#To refresh your knowledge about mlr3 you can take a look at https://mlr3book.mlr-org.com/basics.html.

library(mlbench)
library(mlr3)
library(mlr3learners)
library(mlr3viz)
library(ggplot2)
library(gridExtra)

set.seed(401)

learners <- list(
  # Softmax regression
  softmax_learner = lrn("classif.multinom",
                        trace = FALSE,
                        predict_type = "prob"
  ),
  # K-nearest neighbors
  knn_learner = lrn("classif.kknn",
                    k = 50,
                    predict_type = "prob"
  ),
  # Linear discriminant analysis
  lda_learner = lrn("classif.lda",
                    predict_type = "prob"
  ),
  # Quadratic discriminant analysis
  qda_learner = lrn("classif.qda",
                    predict_type = "prob"
  ),
  nb_learner = lrn("classif.naive_bayes",
                   predict_type = "prob"
  )
)

spirals <- data.frame(mlbench.spirals(n = 500, sd = 0.11))
ggplot(spirals, aes(x.1, x.2, color = classes)) + geom_point()
spirals_task <- TaskClassif$new(id = "spirals", backend = spirals, target = "classes")
ggplot_list <- lapply(
  learners,
  function(learner) plot_learner_prediction(learner, spirals_task) +
    theme_minimal(base_size = 10) + guides(alpha = "none", shape = "none") +
    ggtitle(learner$id)
)

do.call(grid.arrange, ggplot_list)