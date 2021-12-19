# xgb vs lightgbm: prediction times

The code in R/predict_gender.R takes in a simple insurance claims dataset, does minimal data prep, then builds a classification model to predict gender, using both xgb and lgbm. Both models are built using 1000 trees, with no limits on depth beyond default parameters. learning rates are set equal, and each model builds 1000 trees. 

The aim of this isn't to compare performance of models, but rather to compare prediction times. The below table shows the xgboost builds a much more complex model, but generates predictions over test data in a fraction of the time. 
 
|                           | XGB         | LGBM        |
| -----------               | ----------- | ----------- |
| Avg prediction time (ms)  | 1.5         | 60          |
| Number of Splits          | 47,735      | 30,000      |

Based on this result, it looks like lgbm would make for a poor choice of production model, as response times would be drastically slower. Does this generalise to other datasets/objective types etc too? 

The code was run on a 7th gen i3, so minimal gains would be offered through parallel processing. This result probably gives a good reflection of how each algorithm would perform on a single core.

