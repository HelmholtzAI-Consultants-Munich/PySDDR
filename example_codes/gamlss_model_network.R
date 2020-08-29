source("setup_packages.R")
source("dro.R")
library(mgcv)
library(Matrix)
library(gamlss)
library(gamlss.dist)
library(gamlss.add)
saver <- function(x, name) 
  write.table(x, file=paste0("../example_data/gamlss/",name,".csv"),
            row.names = FALSE, col.names = FALSE, sep = ";")

set.seed(13313)

n <- 1000
p <- 4
X <- matrix(runif(n*p, min = -1, max = 1), ncol = p)
saver(X, "X")
true_expectation = -0.8 + X[,1]^2 - X[,2]
true_std = 0.3 + X[,3]*0.5 + sin(4*X[,4])

# sample from a location-scale parametrized Logistic distribution
Y <- rLO(n = n, mu = true_expectation, sigma = exp(true_std))
saver(Y, "Y")


# define data
data = data.frame(Y = Y, X = X)

# fit gamlss model
mod_gamlss <- gamlss(formula = Y ~ ga( ~ 1 + s(X.1) + s(X.2)), 
                     sigma.formula = ~ ga( ~ 1 + s(X.3) + s(X.4)), 
                     family = "LO", 
                     data = data)

# get evaluated basis matrix
s_mu <- lapply(list(s(X.1), s(X.2)), function(t) 
  smoothCon(object = t, data = data))
s_sigma <- lapply(list(s(X.3), s(X.4)), function(t) 
  smoothCon(object = t, data = data))

# design matrices
B_mu <- c(list(1), lapply(s_mu, function(st) st[[1]]$X))
B_sigma <- c(list(1), lapply(s_sigma, function(st) st[[1]]$X))

# penalty matrices
P_mu <- c(list(0), lapply(s_mu, function(st) st[[1]]$S[[1]]))
P_sigma <- c(list(0), lapply(s_sigma, function(st) st[[1]]$S[[1]]))

# set degrees-of-freedom for smooths
df = 4

# get penalties via DRO (see Ruegamer et al., 2020 for explanation)
lambdas_mu <- c(0, sapply(1:2, function(i) DRO(X = B_mu[[1+i]], 
                                               dmat = P_mu[[1+i]], 
                                               df = df)[2])
)

lambdas_sigma <- c(0, sapply(1:2, function(i) DRO(X = B_sigma[[1+i]], 
                                                  dmat = P_sigma[[1+i]], 
                                                  df = df)[2])
)
                   
# construct full design and penalty matrices
B_mu <- do.call("cbind", B_mu)
B_sigma <- do.call("cbind", B_sigma)
P_mu <- Matrix::bdiag(lapply(1:3, function(i) lambdas_mu[i]/n * P_mu[[i]]))
P_sigma <- Matrix::bdiag(lapply(1:3, function(i) lambdas_sigma[i]/n * P_sigma[[i]]))

saver(B_mu, "B_mu")
saver(B_sigma, "B_sigma")
saver(as.matrix(P_mu), "P_mu")
saver(as.matrix(P_sigma), "B_sigma")

# network definition
deep_gamlss_model <- function(ncolX1, ncolX2, distribution, P1, P2)
{
  
  # input for X1
  inpXmu <- layer_input(shape = list(ncolX1))
  # input for X2
  inpXsigma <- layer_input(shape = list(ncolX2))
  
  ### outputs
  # linear net mu
  outXmu <- inpXmu %>% layer_dense(units = 1L, 
                                   use_bias = F, 
                                   activation = "linear", 
                                   name = "linear_layer_mu",
                                   # here we define the penalty of the spline
                                   kernel_regularizer = function(x)
                                   {
                                     k_dot(tf$transpose(x), k_dot(
                                       tf$constant(as.matrix(P1), dtype = "float32"), x)
                                     )
                                   }
  )
  
  # linear net sigma
  outXsigma <- inpXsigma %>% layer_dense(units = 1L, 
                                         use_bias = F, 
                                         activation = "linear", 
                                         name = "linear_layer_sigma",
                                         # here we define the penalty of the spline
                                         kernel_regularizer = function(x)
                                         {
                                           k_dot(tf$transpose(x), k_dot(
                                             tf$constant(as.matrix(P2), dtype = "float32"), x)
                                           )
                                         }
  ) %>% 
    layer_lambda(f = tf$exp)
  
    
  # we add an distributional layer to get the log-likelihood of a GLM
  distOut <- layer_concatenate(list(outXmu, outXsigma)) %>% 
    layer_distribution_lambda(
      make_distribution_fn = function(x)
        distribution(x[,1,drop=F], x[,2,drop=F])
    )
  
  inp <- list(inpXmu, inpXsigma)
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
net <- deep_gamlss_model(ncol(B_mu),
                         ncol(B_sigma),
                         distribution = tfd_logistic,
                         P1 = P_mu,
                         P2 = P_sigma
)


hist <- net %>% fit(
  y =  tf$constant(Y, dtype="float32"),
  x = list(B_mu, B_sigma),
  batch_size       = n, 
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
bcoef_mu <- as.matrix(net$get_layer("linear_layer_mu")$weights[[1]] + 0)
bcoef_sigma <- as.matrix(net$get_layer("linear_layer_sigma")$weights[[1]] + 0)

saver(bcoef_mu, "coef_mu")
saver(bcoef_mu, "coef_sigma")

# create the fitted effects
spline1_NN <- B_mu[,2:11]%*%(bcoef_mu[2:11,])
spline2_NN <- B_mu[,12:21]%*%(bcoef_mu[12:21,])

spline3_NN <- B_sigma[,2:11]%*%(bcoef_sigma[2:11,])
spline4_NN <- B_sigma[,12:21]%*%(bcoef_sigma[12:21,])

# for plotting
r_mu <- range(c(spline1_NN, spline2_NN))
r_sigma <- range(c(spline3_NN, spline4_NN))

# compare
par(mfrow=c(2,4))
term.plot(mod_gamlss, what = "mu")
term.plot(mod_gamlss, what = "sigma")
plot(sort(X[,1]), spline1_NN[order(X[,1])], type="l", ylim = r_mu)
plot(sort(X[,2]), spline2_NN[order(X[,2])], type="l", ylim = r_mu)
plot(sort(X[,3]), spline3_NN[order(X[,3])], type="l", ylim = r_sigma)
plot(sort(X[,4]), spline4_NN[order(X[,4])], type="l", ylim = r_sigma)

