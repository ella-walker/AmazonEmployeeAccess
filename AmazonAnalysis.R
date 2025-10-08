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
  step_dummy(all_nominal_predictors())

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


