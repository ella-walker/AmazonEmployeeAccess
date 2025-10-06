# Useful libraries for this analysis
library(vroom)
library(recipes)
library(tidyverse)
library(tidymodels)
library(ggmosaic)

# Read in test and training data
amazon <- vroom("~/Documents/STAT 348/AmazonEmployeeAccess/archive/train.csv")
testData <- vroom("~/Documents/STAT 348/AmazonEmployeeAccess/archive/test.csv")

##
### Recipe
## Create a recipe that does dummy variable encoding for all nominal predictors
amazon_recipe <- recipe(ACTION ~ ., data = amazon) |>
  step_mutate_at(all_predictors(), fn = factor) |>
  step_other(all_nominal_predictors(), threshold = .001) |>
  step_dummy(all_nominal_predictors())

bake(prep(amazon_recipe), amazon)

##
### EDA
## Create at least 2 exploratory plots for the amazon data

ggplot(data = amazon) +
  geom_mosaic(aes(x = ACTION, fill = ROLE_DEPTNAME))
