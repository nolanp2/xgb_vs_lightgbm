library(vtreat)
library(Matrix)
library(xgboost)
library(lightgbm)
library(plotly)
library(MLmetrics)
library(microbenchmark)

# load and treat data -----------------------------------------------------

#random insurance data, originally came from below url:
# main_data <-
#   read.csv(url("http://dyzz9obi78pm5.cloudfront.net/app/image/id/560ec66d32131c9409f2ba54/n/Auto_Insurance_Claims_Sample.csv"))
main_data <- readRDS("datasets/insurance_data.RDS")

#we'll set gender as the response to predict
resp <- main_data$Gender 
resp <- as.numeric(resp == "M")
main_data$Gender <- NULL

#minimal data prep, throw away high/low variance variables and OHE: 
treatments <- 
  vtreat::designTreatmentsZ(
    dframe = main_data,
    varlist = colnames(main_data),
    minFraction = 0.1,
    rareCount = 100
  )
model_data <- prepare(treatments,main_data)
sparse_data <- Matrix::sparse.model.matrix(~.-1, model_data)

#train/test split: 
set.seed(315)
train_ind = sample(1:nrow(model_data),0.8*nrow(model_data),replace = F)
train_data <- sparse_data[train_ind,]
train_resp <- resp[train_ind]
test_data <- sparse_data[-train_ind,]
test_resp <- resp[-train_ind]

# xgb ---------------------------------------------------------------------


xgb_train <- xgboost::xgb.DMatrix(train_data,label = train_resp)
xgb_test <- xgboost::xgb.DMatrix(test_data,label = test_resp)
xgb = xgboost(data = xgb_train,objective = 'binary:logistic',nrounds = 1000,max_depth = 1000)

# lgbm --------------------------------------------------------------------


lgbm_train <- lightgbm::lgb.Dataset(train_data,label = train_resp)
lgbm_test <- test_data
lgb <- lightgbm(lgbm_train,nrounds = 1000,params = list(objective = "binary"))


# predict -----------------------------------------------------------------

#xgb is way bigger, yet predictions are way faster. 
#at the same time, xgb model is very poor compared to lgb.

#mean times: 1.5ms for xgb, 65ms for lgbm
microbenchmark(px = predict(xgb,xgb_test))
microbenchmark(pl = predict(lgb,lgbm_test))

# xgb is much bigger, with more splits and leaves
object.size(xgb)/1e6
object.size(lgb)/1e6

#lgbm gets a perfect score, xgb is terrible (33.6%)
MLmetrics::Gini(y_pred = px,y_true = test_resp)
MLmetrics::Gini(y_pred = pl,y_true = test_resp)

# plot:lgb ----------------------------------------------------------------

var1 = pl[test_resp == 0]
var1 = density(var1)

var2 = pl[test_resp == 1]
var2 = density(var2,bw = var1$bw)

pltdf <- cbind.data.frame(x = var1$x,y1 = var1$y,y2 = var2$y)

pltdf %>%
  plot_ly %>%
  add_trace(x = ~x, y = ~y1, mode = "lines", fill = "tozeroy", name = "actual = 0",type = 'scatter') %>%
  add_trace(x = ~x, y = ~y2, mode = "lines", fill = "tozeroy", name = "actual = 1",type = 'scatter') %>%
  layout(title = "lgbm prediction density")

# plot: xgb ---------------------------------------------------------------

var1 = px[test_resp == 0]
var1 = density(var1)

var2 = px[test_resp == 1]
var2 = density(var2,bw = var1$bw)

pltdf <- cbind.data.frame(x = var1$x,y1 = var1$y,y2 = var2$y)

pltdf %>%
  plot_ly %>%
  add_trace(x = ~x, y = ~y1, mode = "lines", fill = "tozeroy", name = "actual = 0",type = 'scatter') %>%
  add_trace(x = ~x, y = ~y2, mode = "lines", fill = "tozeroy", name = "actual = 1",type = 'scatter')  %>% 
  layout(title = "xgb prediction density")


# trees -------------------------------------------------------------------


xgbtrees = xgboost::xgb.model.dt.tree(feature_names = xgb$feature_names,xgb)
lgbtrees = lightgbm::lgb.model.dt.tree(lgb)


sum(!is.na(xgbtrees$Split))
sum(!is.na(lgbtrees$split_index))
