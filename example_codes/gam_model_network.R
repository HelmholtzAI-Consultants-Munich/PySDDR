library(mgcv)
library(Matrix)

set.seed(13313)

n <- 1000
X <- matrix(runif(n*2, -2, 2),ncol=2)
write.table(X, file="../example_data/simple_gam/X.csv",
          row.names = FALSE, col.names = FALSE, sep = ";")
true_expectation = 1 + X[,1]^2 - X[,2]
Y <- rpois(n = n, lambda = exp(true_expectation))
write.table(Y, file="../example_data/simple_gam/Y.csv",
          row.names = FALSE, col.names = FALSE, sep = ";")

mod_gam <- gam(Y ~ 1 + s(X[,1]) + s(X[,2]), family="poisson")
# get evaluated basis matrix
B <- model.matrix(mod_gam)
write.table(B, file="../example_data/simple_gam/B.csv",
          row.names = FALSE, col.names = FALSE, sep = ";")

# and penalty matrix
P_gam <- bdiag(c(0, lapply(1:2, function(i) 
  mod_gam$sp[i]*mod_gam$smooth[[i]]$S[[1]]/n)))
write.table(as.matrix(P_gam), file="../example_data/simple_gam/full_P.csv",
          row.names = FALSE, col.names = FALSE, sep = ";")
P_only <- bdiag(c(0, lapply(1:2, function(i) 
  mod_gam$smooth[[i]]$S[[1]])))
write.table(as.matrix(P_only), file="../example_data/simple_gam/P_only.csv",
          row.names = FALSE, col.names = FALSE, sep = ";")
sps <- mod_gam$sp/n
write.table(as.matrix(sps), file="../example_data/simple_gam/smoothing_parameters.csv",
          row.names = FALSE, col.names = FALSE, sep = ";")

# true coefficients
gam_coef <- coef(mod_gam)
write.table(gam_coef, file="../example_data/simple_gam/true_coefficients.csv",
          row.names = FALSE, col.names = FALSE, sep = ";")

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
deep_linear_model <- function(ncolX, distribution, P)
{
  
  # input for X
  inpX <- layer_input(shape = list(ncolX))
  
  ### outputs
  # linear net
  outX <- inpX %>% layer_dense(units = 1L, 
                               use_bias = F, 
                               activation = "linear", 
                               name = "linear_layer",
                               # here we define the penalty of the spline
                               kernel_regularizer = function(x)
                               {
                                   k_dot(tf$transpose(x), k_dot(
                                     tf$constant(as.matrix(P), dtype = "float32"), x)
                                   )
                               }
  )

  
  # we add an distributional layer to get the log-likelihood of a GLM
  distOut <- outX %>% 
    layer_distribution_lambda( # feed transformed linear predictor into 
      # distribution layer
      function(x)
        distribution(x[, 1, drop = FALSE])
    )
  
  inp <- list(inpX)
  mod <- keras_model(inp, distOut)
  
  ind_fun <- function(x) tfd_independent(x,1)
  
  negloglik <- function(y, model){
    - 1 * (model %>% ind_fun %>% tfd_log_prob(y))
  }
  
  mod %>% compile(
    optimizer = optimizer_rmsprop(),
    loss      = negloglik
  )
  
  return(mod)
  
}


# initializes network with
# distribution = Poisson
# response function = exp
net <- deep_linear_model(ncol(B), # we here provide the basis evaluated data
                         # use the poisson distribution layer
                         # as an example for a GLM
                         distribution = function(x) 
                           tfd_poisson(log_rate = x),
                         P = P_gam
)


hist <- net %>% fit(
  y =  tf$constant(Y, dtype="float32"),
  x = list(B),
  batch_size       = 1000, 
  epochs           = 2500,
  validation_split = 0.2,
  view_metrics = FALSE,
  verbose = TRUE,
  callbacks = list(
    # print_weights, 
    callback_early_stopping(patience = 100)
  )
)

# check the weigths of the network -> in this case the basis coefficients
(bcoef <- as.matrix(net$get_layer("linear_layer")$weights[[1]] + 0))

# create the fitted effects
spline1_NN <- B[,2:10]%*%(bcoef[2:10,])
spline2_NN <- B[,11:19]%*%(bcoef[11:19,])

plot(mod_gam, select=1)
points(sort(X[,1]), spline1_NN[order(X[,1])], type="l", col="red")
plot(mod_gam, select=2)
points(sort(X[,2]), spline2_NN[order(X[,2])], type="l", col="red")

