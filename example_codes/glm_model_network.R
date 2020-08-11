set.seed(13313)

n <- 100
X <- matrix(rnorm(n*2),ncol=2)
true_expectation = X[,1]*2 - 1*X[,2]
Y <- rpois(n = n, lambda = exp(true_expectation))

# install.packages("reticulate")
# reticulate::install_miniconda()

# install.packages("tensorflow")
library(tensorflow)
# install_tensorflow(version = "2.0")
# install.packages("keras")
# not sure if needed: install_keras()
library(keras)
# install.packages("tfprobability")
# install_tfprobability(version = "0.8", tensorflow = "2.0")
library(tfprobability)
# catch start up error
try(a <- tfd_normal(0,1))

# network definition
deep_linear_model <- function(ncolX, distribution, response_fun)
{
  
  # input for X
  inpX <- layer_input(shape = list(ncolX))
  
  ### outputs
  # linear net
  outX <- inpX %>% layer_dense(units = 1L, 
                               use_bias = F, 
                               activation = "linear", 
                               name = "linear_layer")
  
  # we add an distributional layer to get the log-likelihood of a GLM
  distOut <- outX %>% 
    layer_lambda(f = response_fun) %>% # transform with response fun
    layer_distribution_lambda( # feed transformed linear predictor into 
      # distribution layer
      function(x)
        distribution(x[, 1, drop = FALSE])
      )
  
  inp <- list(inpX)
  mod <- keras_model(inp, distOut)
  
  negloglik <- function(y, model){
    - 1 * (model %>%  tfd_log_prob(y))
  }
  
  mod %>% compile(
    optimizer = optimizer_adadelta(),
    loss      = negloglik
  )
  
  return(mod)
  
}

# initializes network with
# distribution = Poisson
# response function = exp
net <- deep_linear_model(ncol(X), 
                         # use the poisson distribution layer
                         # as an example for a GLM
                         distribution = function(x) tfd_poisson(rate = x),
                         # with response function exp
                         response_fun = function(x)
                         {
                           1e-8 + # for stability 
                           tf$math$exp(x)
                         }
                         )

# callback for weights
print_weights <- keras::callback_lambda(
  on_epoch_end = function(batch, logs = list()) {
    # print("After")
    print(net$get_layer("linear_layer")$weights[[1]])
  }
)


hist <- net %>% fit(
  y =  tf$constant(Y, dtype="float32"),
  x = list(X),
  batch_size       = 100, 
  epochs           = 2500,
  validation_split = 0.2,
  view_metrics = FALSE,
  verbose = TRUE,
  callbacks = list(
    print_weights, 
    callback_early_stopping(patience = 50)
    )
)

# check the weigths of the network
net$get_layer("linear_layer")$weights[[1]]

# what would the glm say?
coef(glm(Y ~-1+X, family = "poisson"))
