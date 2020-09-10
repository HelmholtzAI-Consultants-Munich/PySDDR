set.seed(13313)

n <- 100
X <- matrix(rnorm(n*2),ncol=2)
Y <- X[,1]*2 - 1*X[,2] + rnorm(n)
  
# install.packages("reticulate")
# reticulate::install_miniconda()

# install.packages("tensorflow")
library(tensorflow)
# install_tensorflow(version = "2.0")
# install.packages("keras")
# not sure if needed: install_keras()
library(keras)

# network definition
deep_linear_model <- function(ncolX)
{
  
  # input for X
  inpX <- layer_input(shape = list(ncolX))

  ### outputs
  # linear net
  outX <- inpX %>% layer_dense(units = 1, use_bias = F, 
                               activation = "linear", 
                               name = "linear_layer")
  
  inp <- list(inpX)
  mod <- keras_model(inp, outX)
  
  mod %>% compile(
    optimizer = optimizer_adam(),
    loss      = loss_mean_squared_error
  ) 
  
}

# initializes network
net <- deep_linear_model(ncol(X))

# callback for weights
print_weights <- keras::callback_lambda(
  on_epoch_end = function(batch, logs = list()) {
    # print("After")
    print(net$get_layer("linear_layer")$weights[[1]])
  }
)


hist <- net %>% fit(
  y = Y,
  x = list(X),
  batch_size       = 100, 
  epochs           = 1000,
  validation_split = 0.2,
  view_metrics = TRUE,
  verbose = FALSE,
  callbacks = list(#print_weights, 
                   callback_early_stopping(patience = 50)),
  shuffle = FALSE
)

# check the weigths of the network
net$get_layer("linear_layer")$weights[[1]]

# what would the lm say?
coef(lm(Y~-1+X))
