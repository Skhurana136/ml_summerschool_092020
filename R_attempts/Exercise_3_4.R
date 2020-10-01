data <- data.frame(
  ID = c(1,2,3,4,5,6,7,8,9,10),
  Actual_Class = c("0","0","1","1","1","0","1","1","0","0"),
  Score = c(0.33, 027, 0.11, 0.38, 0.17, 0.63, 0.62, 0.33, 0.15, 0.57)
)

#a) Create a confusion matrix assuming the decision boundary at 0.5.
#logistic regression on the whole task:
learner <- lrn("classif.log_reg", predict_type = "prob")
data_task <- TaskClassif$new (id = "logreg", backend = data[,c("Score", "Actual_Class")], target = "Actual_Class")

learner$train(task = data_task)
#Predict model
model_prediction <-learner$predict(task = data_task)
model_prediction$set_threshold(0.5)
model_prediction$confusion


#b) Calculate: precision, sensitivity, negative predictive value, specificity, accuracy, error rate and F-measure.

#c) Draw the ROC curve and interpret it. Feel free to use R for the drawing.

#d) Calculate the AUC.

