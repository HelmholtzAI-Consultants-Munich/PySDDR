# PySDDR

## What is this?

_Short intro on PySDDR and a figure which shows how it works? Would be good if we had a real use case example with DL being useful_


PySDDR is a python package used for regression tasks which combines statistical regression models and neural networks and takes advantage of the strengths of each to deal with multi-modal data.

It was built based on the concepts presented in _paper_ and follows the R implementation found here _link_. 

The package works for both mean and distributional regression while assuming a distribution of any given data, as long as each parameter is defined by _a linear predictor_. 

Mention ortogonalization as that is the main contribution!

_David's intro_
deepregression is an implementation of a large number of statistical regression models, but fitted in a neural network. It can be used for mean regression as well as for distributional regression, i.e. estimating any parameter of the assumed distribution, not just the mean. Each parameter can be defined by a linear predictor. deepregression uses the pre-processing of the package mgcv to build smooth terms and has a similar formula interface as mgcv::gam. As all models are estimated in a neural network, deepregression can not only make use of TensorFlow as a computing engine, but allows to specify parameters also by additional deep neural networks (DNNs). This allows to include, e.g., CNNs or LSTMs into the model formula and thus incorporate unstructured data sources into a regression model. When combining structured regression models with DNNs, the software uses an orthogonalization cell to make the structured parts of the model formula (the linear and smooth terms) identifiable in the presence of the DNN(s).

## Installation

To install the necessary packages for this framework run:

```
pip install -r requirements.txt
```

If you are using conda first install pip by: ```conda install pip```

## Model

### SDDRNet

The model architecture is built dynamically, depending on the user inputs such as the assumed distribution and _linear predictors_. It combines statistical models and neural networks into one larger network, namely SDDRNet, responsible for integrating all parts. Depending on the number of parameters of the assumed distribution defined by the user, SDDRNet consists of _a OR the same_ number of smaller networks, SDDR_Param_Net, which are built in parallel. A formula is given by the user for each parameter based on which each SDDR_Param_Net is built and the output of each is the predicted parameter value. Within SDDRNet, these are collected, normalized based on the distrubution's rules, and given as input to a distributional layer. From the distributional layer a log loss is computed, to which a _regularization_ term is added to compute the final loss which is then backpropagated. An example of this can be seen below.


# SDDR_Param_Net

As mentioned, each SDDR_Param_Net predicts a parameter of the distribution. Depending on the formula the user has given for this parameter the SDDR_Param_Net is built. The inputs to the network are the linear part of the data, the processed structured data and the unstructured data. _when does orthog of linear and structured happen-mention!_ The processed structured data has already been fitted with splines _word this better_, the number of which is defined by the user in the equation. The outputs of the smoothing _terms_ are concatenated and given to a linear layer, which we name the Structured Head. The unstructured data is given into one or multiple neural networks. Both the number and arcitecture of these are pre-defined by the user and are built within the SDDR_Param_Net in a parallel fashion. Their outputs are concatenated and given to the orthogonalization layer. 

## User inputs
_Here discuss and explain all the inputs the user must give_
### Data
_How data can be inputted_ 
### Distributions
_List of available distributions and params_
## Deep Neural Networks
_How a NN can be defined, e.g. by torch instances directly, loaded from script_

.

.

.

## Features
### Scientific features
_Here discuss the scientific features available to the user, e.g. w or w/o orthogonolization, optimizer options etc._
### Run/training features
<find a better name for run and training features:P>
_Here discuss the options available to the user while training, e.g. loading pretrained weights_

## Examples of using scientific features
```
this is an example, here we can link also to test_run.ipynb
```
_We should however add more content than currently in jupyter notebook_

 ## Examples of training features
```
this is an example, here we can link also to example_usage.ipynb
```


## Results
_Would be nice if eventually we have some nice results to show, though this is somehow similar to test_run.ipynb_
