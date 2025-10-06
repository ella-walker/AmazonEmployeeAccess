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
amazon <- vroom("~/Documents/STAT 348/AmazonEmployeeAccess/archive/train.csv")
testData <- vroom("~/Documents/STAT 348/AmazonEmployeeAccess/archive/test.csv")

##
### EDA
## Create at least 2 exploratory plots for the amazon data

skim(amazon)
glimpse(amazon)
plot_histogram(amazon)

amazon$ACTION <- as.factor(amazon$ACTION)
amazon$ROLE_DEPTNAME <- as.factor(amazon$ROLE_DEPTNAME)


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


