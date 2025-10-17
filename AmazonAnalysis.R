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
  step_other(all_nominal_predictors(), threshold = .001) |>
  step_lencode_glm(all_nominal_predictors(), outcome = vars(ACTION)) |>
  step_normalize(all_predictors())

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
            file = "~/Documents/STAT 348/AmazonEmployeeAccess/logistic_regression.csv",
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
            file = "~/Documents/STAT 348/AmazonEmployeeAccess/penalized_logistic.csv",
            delim = ",")


##
### Random Forest
##

random_forest_mod <- rand_forest(mtry = tune(),
                                 min_n = tune(),
                                 trees = 500) |>
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
            file = "~/Documents/STAT 348/AmazonEmployeeAccess/random_forest.csv",
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

vroom_write(random_forest_preds,
            file = "~/Documents/STAT 348/AmazonEmployeeAccess/knn.csv",
            delim = ",")
