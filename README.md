# PySDDR

## What is this?

PySDDR is a python package used for regression tasks which combines statistical regression models and neural networks and takes advantage of the strengths of each to deal with multi-modal data.

It was built based on the concepts presented in _paper_ and follows the R implementation found here _link_. One of the main contributions of both the paper and packages is the introduction of an orthogonalization layer to help identifiability between structured and unstructured data.

The package works for both mean and distributional regression while assuming a distribution of any given data, as long as each parameter is defined by _a linear predictor_. It is built in such a way that allows for the beginner to easily use and take advantage of this unifying framework but also enables the more advanced user to exploit features and tweak parameters in a flexible and interactive way. The main model is built in a dynamic way depending on the user input and can accept any number of neural networks, from the simplest classifier to the more complicated architectures, such as LSTMs, integrating all these in a unified network written in PyTorch. Meanwhile the structured data is smoothed using splines as basis function and its partial effects can be visualized during evaluation.

## Installation

To install the necessary packages for this framework run:

```
pip install -r requirements.txt
```

If you are using conda first install pip by: ```conda install pip```

## Model


### SDDRNet

The model architecture is built dynamically, depending on the user inputs such as the assumed distribution and _linear predictors_. It combines statistical models and neural networks into one larger unifying network, namely SDDRNet, responsible for integrating all parts. Depending on the number of parameters of the assumed distribution defined by the user, SDDRNet consists of _a OR the same_ number of smaller networks, SDDR_Param_Net, which are built in parallel. A formula is given by the user for each parameter based on which each SDDR_Param_Net is built and the output of each is the predicted parameter value. Within SDDRNet, these are collected, normalized based on the distrubution's rules, and given as input to a distributional layer. From the distributional layer a log loss is computed, to which a _regularization_ term is added forming the final loss which is then backpropagated. SDDRNet accepts the data after preprocessing has been applied. An example of this can be seen below.

![image](https://github.com/davidruegamer/PySDDR/blob/prepare_data_feature_branch/sddr_net.jpg)

### Preprocessing

PySDDR has been built to accept data in many types and forms, e.g. sequential data, imaging data etc., both singly and combined. The individual features of these data can be of three types: linear, structured and unstructured. The user needs to define which features belong to each pf these three types and this is done through the parameters' equations. The first part of preprocessing is then to split the input data into these three types depending on the given formulas. Additionally, the structured part is fitted to one or many basis functions giving smooth partial effects. Currently for the basis functions, b-splines are used per default, where Cyclic Cubic splines are also available.

### SDDR_Param_Net

As mentioned, each SDDR_Param_Net predicts a parameter of the distribution. Depending on the formula the user has given for this parameter the SDDR_Param_Net is built. The inputs to the network are the linear part of the data, the processed structured data and the unstructured data. _when does orthog of linear and structured happen-mention!_ The processed structured data has already been fitted with splines _word this better_, the number of which is defined by the user in the equation. The outputs of the smoothing _terms_ are concatenated and given to a fully connected _(linear)_ layer, which we name Structured Head. The unstructured data is given into one or multiple neural networks. Both the number and arcitecture of these are pre-defined by the user and are built within the SDDR_Param_Net in a parallel fashion. Their outputs are concatenated and together with the processed structured data are given to the orthogonalization layer. The orthogonalization layer ensures identifiability of the structured data by removing structured covariates from the neural network outpus if they are present. The orthogonalized, concatenated output of the neural networks is fed into a fully connected _(linear)_ layer, which we name Deep Head. The sum of this output and the output of the Structured Head forms the paramter prediction of the SDDR_Param_Net. An example of the architecture can be seen below.

![image](https://github.com/davidruegamer/PySDDR/blob/prepare_data_feature_branch/sddr_param_net.jpg)


## User inputs

_Here discuss and explain all the inputs the user must give_

### Data

_How data can be inputted_ 

### Distributions

_List of available distributions and params_

### Deep Neural Networks
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
