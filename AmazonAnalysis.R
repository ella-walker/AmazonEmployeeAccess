# Useful libraries for this analysis
library(vroom)
library(recipes)
library(tidyverse)
library(tidymodels)
library(ggmosaic)
library(skimr)
library(dplyr)
library(DataExplorer)
library(corrplot)
library(embed)
library(discrim)
library(kernlab)
library(themis)

# Read in test and training data
amazon <- vroom("~/Documents/STAT 348/AmazonEmployeeAccess/archive/train.csv") |>
  mutate(ACTION = factor(ACTION))
testData <- vroom("~/Documents/STAT 348/AmazonEmployeeAccess/archive/test.csv") |>
  mutate(Id = row_number())

##
### EDA
## Create at least 2 exploratory plots for the amazon data

skim(amazon)
glimpse(amazon)
plot_histogram(amazon)

ggplot(data = amazon) +
  geom_mosaic(aes(x = product(ROLE_DEPTNAME), fill = ACTION))

plot_correlation(amazon)

#To be honest, I'm not sure what plots are best for such a large dataset

##
### Recipe
## Create a recipe that does dummy variable encoding for all nominal predictors
amazon_recipe <- recipe(ACTION ~ ., data = amazon) |>
  step_mutate_at(all_predictors(), fn = factor) |>
  step_lencode_glm(all_nominal_predictors(), outcome = vars(ACTION)) |>
  step_normalize(all_predictors())

amazon_recipe_pca <- recipe(ACTION ~ ., data = amazon) |>
  step_mutate_at(all_predictors(), fn = factor) |>
  step_other(all_nominal_predictors(), threshold = .001) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_predictors()) |>
  step_pca(all_predictors(), threshold = .8)

bake(prep(amazon_recipe), amazon)


##
### Logistic Regression
##

## Set Model
logRegModel <- logistic_reg() |> 
  set_engine("glm")

## Set Workflow
logReg_workflow <- workflow() |>
  add_recipe(amazon_recipe) |>
  add_model(logRegModel) |>
  fit(data = amazon)

## Prediction
logistic_predictions <- predict(logReg_workflow,
                                new_data = testData,
                                type = "prob")

## Kaggle Formatting
kaggle_submission <- bind_cols(testData["Id"], logistic_predictions) |>
  rename(Action = .pred_1) |>
  select(Id, Action)
           
vroom_write(kaggle_submission,
            file = "~/Documents/STAT 348/AmazonEmployeeAccess/logistic_regression_pca.csv",
            delim = ",")


##
### Penalized Regression 
##

penalized_mod <- logistic_reg(mixture= tune(), penalty = tune()) |>
  set_engine("glmnet")

penalized_wf <- workflow() |>
  add_recipe(amazon_recipe) |>
  add_model(penalized_mod)

penalized_tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

folds <- vfold_cv(amazon, v = 5, repeats = 1)

penalized_CV_results <- penalized_wf |>
  tune_grid(resamples = folds,
            grid = penalized_tuning_grid,
            metrics = metric_set(roc_auc))

penalized_bestTune <- penalized_CV_results |>
  select_best(metric = "roc_auc")

penalized_final_wf <-
  penalized_wf |>
  finalize_workflow(penalized_bestTune) |>
  fit(data = amazon)

penalized_preds <- penalized_final_wf |>
  predict(new_data = testData, type = "prob") |>
  bind_cols(testData) |>
  rename(ACTION = .pred_1) |>
  select(id, ACTION)

vroom_write(penalized_preds,
            file = "~/Documents/STAT 348/AmazonEmployeeAccess/penalized_logistic_pca.csv",
            delim = ",")


##
### Random Forest
##

random_forest_mod <- rand_forest(mtry = tune(),
                                 min_n = tune(),
                                 trees = 1000) |>
  set_engine("ranger") |>
  set_mode("classification")

random_forest_wf <- workflow() |>
  add_recipe(amazon_recipe) |>
  add_model(random_forest_mod)

random_forest_grid_of_tuning_params <- grid_regular(mtry(range = c(1, ncol(amazon) - 1)),
                                                    min_n(),
                                                    levels = 5)

folds <- vfold_cv(amazon, v = 5, repeats = 1)

random_forest_CV_results <- random_forest_wf |>
  tune_grid(resamples = folds,
            grid = random_forest_grid_of_tuning_params,
            metrics = metric_set(roc_auc))

random_forest_bestTune <- random_forest_CV_results |>
  select_best(metric = "roc_auc")

random_forest_final_wf <- random_forest_wf |>
  finalize_workflow(random_forest_bestTune) |>
  fit(data = amazon)

random_forest_preds <- predict(random_forest_final_wf, new_data = testData)

random_forest_preds <- random_forest_final_wf |>
  predict(new_data = testData, type = "prob") |>
  bind_cols(testData) |>
  rename(ACTION = .pred_1) |>
  select(id, ACTION)

vroom_write(random_forest_preds,
            file = "~/Documents/STAT 348/AmazonEmployeeAccess/random_forest_new.csv",
            delim = ",")

##
### KNN
##

knn_model <- nearest_neighbor(neighbors = tune()) |>
  set_mode("classification") |>
  set_engine("kknn")

knn_wf <- workflow() |>
  add_recipe(amazon_recipe) |>
  add_model(knn_model)

knn_grid_of_tuning_params <- grid_regular(
  neighbors(range = c(1, 25)),
  levels = 10
)


folds <- vfold_cv(amazon, v = 5, repeats = 1)

knn_CV_results <- knn_wf |>
  tune_grid(resamples = folds,
            grid = knn_grid_of_tuning_params,
            metrics = metric_set(roc_auc))

knn_bestTune <- knn_CV_results |>
  select_best(metric = "roc_auc")

knn_final_wf <- knn_wf |>
  finalize_workflow(knn_bestTune) |>
  fit(data = amazon)

knn_preds <- knn_final_wf |>
  predict(new_data = testData, type = "prob") |>
  bind_cols(testData) |>
  rename(ACTION = .pred_1) |>
  select(id, ACTION)

vroom_write(knn_preds,
            file = "~/Documents/STAT 348/AmazonEmployeeAccess/knn.csv",
            delim = ",")


## 
### Naive Bayes
## 

nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) |>
  set_mode("classification") |>
  set_engine("naivebayes")

nb_wf <- workflow() |>
  add_recipe(amazon_recipe) |>
  add_model(nb_model)

nb_grid_of_tuning_params <- grid_regular(
  Laplace(range = c(0.001,2)),
  smoothness(range = c(0.001,2)),
  levels = 5
)

folds <- vfold_cv(amazon, v = 5, repeats = 1)

nb_CV_results <- nb_wf |>
  tune_grid(resamples = folds,
            grid = nb_grid_of_tuning_params,
            metrics = metric_set(roc_auc))

nb_bestTune <- nb_CV_results |>
  select_best(metric = "roc_auc")

nb_final_wf <- nb_wf |>
  finalize_workflow(nb_bestTune) |>
  fit(data = amazon)

nb_preds <- nb_final_wf |>
  predict(new_data = testData, type = "prob") |>
  bind_cols(testData) |>
  rename(ACTION = .pred_1) |>
  select(id, ACTION)

vroom_write(nb_preds,
            file = "~/Documents/STAT 348/AmazonEmployeeAccess/nb.csv",
            delim = ",")


##
### Neural Networks
##

nn_recipe <- recipe(ACTION ~ ., data = amazon) |>
  step_mutate(across(where(is.character), as.factor)) |>
  step_other(all_nominal_predictors(), threshold = .001) |>
  step_lencode_glm(all_nominal_predictors(), outcome = vars(ACTION)) |>
  step_normalize(all_predictors()) |>
  step_range(all_numeric_predictors(), min = 0, max = 1)

nn_model <- mlp(hidden_units = tune(),
                epochs = 150) |>
  set_engine("keras", verbose=0) |>
  set_mode("classification")

nn_wf <- workflow() |>
  add_recipe(nn_recipe) |>
  add_model(nn_model)


nn_tuneGrid <- grid_regular(hidden_units(range = c(1, 17)),
                            levels = 1)

folds <- vfold_cv(amazon, v = 5)

tuned_nn <- nn_wf |>
  tune_grid(resamples = folds,
            grid = nn_tuneGrid,
            metrics = metric_set(roc_auc))

nn_bestTune <- tuned_nn |>
  select_best(metric = "roc_auc")

nn_final_wf <- nn_wf |>
  finalize_workflow(nn_bestTune) |>
  fit(data = amazon)

nn_preds <- nn_final_wf |>
  predict(new_data = testData, type = "prob") |>
  bind_cols(testData) |>
  rename(ACTION = .pred_1) |>
  select(id, ACTION)

vroom_write(nn_preds,
            file = "~/Documents/STAT 348/AmazonEmployeeAccess/nn.csv",
            delim = ",")

tuned_nn |> collect_metrics() |>
  filter(.metric == "roc_auc") |>
  ggplot(aes(x = hidden_units, y = mean)) + 
  geom_line()


###
#### SVM
###
amazon_recipe_svm <- recipe(ACTION ~ ., data = amazon) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_pca(all_predictors(), threshold=0.99) %>%
  step_downsample(ACTION)

# SVM poly
svm_poly_model <- svm_poly(degree = 1, cost = 0.0131) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_poly_wf <- 
  workflow() %>%
  add_model(svm_poly_model) %>%
  add_recipe(amazon_recipe_svm)

svm_poly_fit <- svm_poly_wf |>
  fit(data = amazon)

poly_preds <- svm_poly_fit |>
  predict(new_data = testData, type = "prob") |>
  bind_cols(testData) |>
  rename(ACTION = .pred_1) |>
  select(id, ACTION)

vroom_write(poly_preds,
            file = "~/Documents/STAT 348/AmazonEmployeeAccess/svm_poly.csv",
            delim = ",")

# SVM Radial
svmRadial <- svm_rbf(rbf_sigma = 0.177, cost = 0.00316) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

radial_wf <- workflow() |>
  add_recipe(amazon_recipe_svm) |>
  add_model(svmRadial)

svm_radial_fit <- radial_wf |>
  fit(data = amazon)

radial_preds <- svm_radial_fit |>
  predict(new_data = testData, type = "prob") |>
  bind_cols(testData) |>
  rename(ACTION = .pred_1) |>
  select(id, ACTION)

vroom_write(radial_preds,
            file = "~/Documents/STAT 348/AmazonEmployeeAccess/svm_radial.csv",
            delim = ",")

# SVM Linear
svmLinear <- svm_linear(cost = 0.0131) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

linear_wf <- workflow() |>
  add_recipe(amazon_recipe_svm) |>
  add_model(svmLinear)

svm_linear_fit <- linear_wf |>
  fit(data = amazon)

linear_preds <- svm_linear_fit |>
  predict(new_data = testData, type = "prob") |>
  bind_cols(testData) |>
  rename(ACTION = .pred_1) |>
  select(id, ACTION)

vroom_write(linear_preds,
            file = "~/Documents/STAT 348/AmazonEmployeeAccess/svm_linear.csv",
            delim = ",")


###
#### Balancing Data
###

smote_recipe <- recipe(ACTION ~ ., data=amazon) |>
  step_mutate_at(all_predictors(), fn = factor) |>
  step_other(all_nominal_predictors(), threshold = .001) |>
  step_lencode_glm(all_nominal_predictors(), outcome = vars(ACTION)) |>
  step_normalize(all_predictors()) |>
  step_smote(all_outcomes(), neighbors=5)

bake(prep(smote_recipe), amazon)

random_forest_mod <- rand_forest(mtry = tune(),
                                 min_n = tune(),
                                 trees = 500) |>
  set_engine("ranger") |>
  set_mode("classification")

random_forest_wf <- workflow() |>
  add_recipe(smote_recipe) |>
  add_model(random_forest_mod)

random_forest_grid_of_tuning_params <- grid_regular(mtry(range = c(1, ncol(amazon) - 1)),
                                                    min_n(),
                                                    levels = 5)

folds <- vfold_cv(amazon, v = 5, repeats = 1)

random_forest_CV_results <- random_forest_wf |>
  tune_grid(resamples = folds,
            grid = random_forest_grid_of_tuning_params,
            metrics = metric_set(roc_auc))

random_forest_bestTune <- random_forest_CV_results |>
  select_best(metric = "roc_auc")

random_forest_final_wf <- random_forest_wf |>
  finalize_workflow(random_forest_bestTune) |>
  fit(data = amazon)

random_forest_preds <- predict(random_forest_final_wf, new_data = testData)

random_forest_preds <- random_forest_final_wf |>
  predict(new_data = testData, type = "prob") |>
  bind_cols(testData) |>
  rename(ACTION = .pred_1) |>
  select(id, ACTION)

vroom_write(random_forest_preds,
            file = "~/Documents/STAT 348/AmazonEmployeeAccess/random_forest_smote.csv",
            delim = ",")
