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

## Tutorials

Two tutorials are available in the [tutorials](https://github.com/davidruegamer/PySDDR/tree/dev/tutorials) directory. One is aimed as a beginner's guide, where toy data is used to familiarize the user with the package and the second includes a more advanced tutorial with an example of applying the package also to unstructured data.

## Contents

[Model](#Model)

[User Interface] (#Sddr-User-Interface)

[Features] (#Features)

[Results] (Results)

## Model


### SddrNet

The model architecture is built dynamically, depending on the user inputs such as the assumed distribution and _linear predictors_. It combines statistical models and neural networks into one larger unifying network, namely SddrNet, responsible for integrating all parts. Depending on the number of parameters of the assumed distribution defined by the user, SDDRNet consists of a number of smaller networks, SddrFormulaNet, which are built in parallel. A formula is given by the user for each parameter based on which each SddrFormulaNet is built and the output of each is the predicted parameter value. Within SddrNet, these are collected, normalized based on the distrubution's rules, and given as input to a distributional layer. From the distributional layer a log loss is computed, to which a smoothing penalty is added for regularization forming the final loss which is then backpropagated. SddrNet accepts the data after preprocessing has been applied. An example of this can be seen below.

![image](https://github.com/davidruegamer/PySDDR/blob/dev/images/sddr_net.jpg)

### Preprocessing

PySDDR has been built to accept tabular and imaging data, both singly and combined. The individual features of these data can be of three types: linear, structured and unstructured. The user needs to define which features belong to each of these three types and this is done through the parameters' formulas. The first part of preprocessing is then to split the input data into these three types depending on the given formulas. Additionally, the structured part is fitted to one or many basis functions giving smooth partial effects. Currently for the basis functions, b-splines are used per default, where Cyclic Cubic splines are also available.

### SddrFormulaNet

As mentioned, each SddrFormulaNet predicts a parameter of the assumed distribution. Depending on the formula the user has given for this parameter the SddrFormulaNet is built. The inputs to the network are the linear part of the data, the processed structured data and the unstructured data. _when does orthog of linear and structured happen-mention!_ The processed structured data has already been fitted with splines _word this better_, the number of which is defined by the user in the equation. The outputs of the smoothing _terms_ are concatenated and given to a fully connected _(linear)_ layer, which we name Structured Head. The unstructured data is given into one or multiple neural networks. Both the number and arcitecture of these are pre-defined by the user and are built within the SddrFormulaNet in a parallel fashion. Their outputs are concatenated and together with the processed structured data are given to the orthogonalization layer. 
The orthogonalized, concatenated output of the neural networks is fed into a fully connected _(linear)_ layer, which we name Deep Head. The sum of this output and the output of the Structured Head forms the parameter prediction of the SddrFormulaNet. An example of the architecture can be seen below.

![image](https://github.com/davidruegamer/PySDDR/blob/dev/images/sddr_param_net.jpg)

### Orthogonalization

Orthogonalization ensures identifiability of the data by a decomposition of covariates corresponding to the linear, structured and unstructured  part of the data.
It occurs in two parts of the network. The first is performed once during preprocessing and only if linear features are a subset of the structured inputs. For example ```spline(x3, bs='bs', df=9, degree=3)``` is orthogonalized with respect to the intercept and x3. If any terms x2, x4 etc. are present they are ignored in this orthogonalization step. The second orthogonalization occurs in every forward step of the network and follows the same principle as before: it only occurs if structured features are a subset of the unstructured inputs. The formula used for the ortogonalization is the same in both cases and can be described as follows:

Assume we have structured data $$X$$ and unstructured data $$U$$ which pass through the deep networks (defined by the user) and concatenated giving latent features $$\hat{U} = d(U)$$. Then we can replace $$\hat{U}$$ with $$\tilde{U} = P_{orthog}\hat{U}$$

For structured head weights $$w$$ and deep head weights $$\gamma$$ the ouptut of the SddrFormulaNet will then be:

$$ \eta = Xw + \tilde{U}\gamma$$


## Sddr User Interface
 
The user interacts with the package through the Sddr class. An overview of this class and its iteraction with the rest of the package can be seen in the fgigure below:



### Training

For training two simple steps are required by the user:

* Initialize an Sddr instance, e.g. ```sddr = Sddr(config=config)```
* Train with the structured training data by ```sddr.train(target, structured_data)```
* Train with the structured and unstructured training data by ```sddr.train(target, structured_data, unstructured_data)```
* Train, plot and save loss curve by ```sddr.train(target, structured_data, unstructured_data, plot=True)```

### Evaluating

The user can then perform an evalutation of the training on any of the parameter's distribution, e.g. for a Poisson distribution: ```sddr.eval('rate') ```. This will return and plot the partial effects of the structured features. To turn off the plot functionality the user must set ```plot=False ``` when calling ```sddr.eval```.

* To get the trained distribution the user can call ```distribution_layer = sddr.get_distribution()```. From this the user can then get all the properties avalaible from [PyTorch's Probability Distributions package](https://pytorch.org/docs/stable/distributions.html) (torch.distributions), e.g. the mean ```distribution_layer.mean``` or the standard deviaton ```distribution_layer.stddev```.
* To get the trained network's weights, i.e. coefficients, for the structured part, the user can call: 

### Saving

After training and evaluation the model can be saved by ```sddr.save()``` or ```sddr.save('MyModel.pth')```, if the user wishes to save the model with a name different than the default _model.pth_. This will be saved in the output directory defined by the user in the config, or if no output directory has been defined an _outputs_ directory is automatically created.

### Predicting

At a later stage the user can again initialize an Sddr instance and use the previously trained model to make predictions on unseen data. This can be done by:

* Initialize an Sddr instance, e.g. ```sddr = Sddr(config=config) ```
* Load model ```sddr.load(model_name, training_data) ``` 
* Predict on unseen linear and structured data ```sddr.predict(data) ```
* Predict on unseen linear, structured data and unstructured data ```sddr.predict(data, unstructured_data) ```
* Predict on unseen data and clip the data range to fall inside training data range if a out of range error occurs: ```sddr.predict(data, clipping=True) ```
* Predict on unseed data and plot a figure for each spline defined in each formula of the distribution's parameters```sddr.predict(data, plot=True) ```

The distrubution as well as the partial effects for all structured features of all parameters of the distribution will be returned. 

Note here that the training data also needs to be provided during load for the preprocessing steps, i.e. basis functions creation, to be performed.

### Resume

The user may also wish to load a pretrained model to resume training. For this the first two steps (init, load) from above need to be performed and then the user can resume training by ```sddr.train(target, structured_data, resume=True)```


### User inputs

There are a number of inputs which need to be defined by the user before training, or testing can be performed. The user can either directly define these in their run script or give all the parameters through a config file - for more information see [Training features](#Training features). 

A list of all inputs the user needs to give can be seen next:

**data:** the input data

**target:** the target data, i.e. our ground truth

**output_dir:** the path of the output directory in which to save training results, such as loss figures or checkpoints  **-> do we need it or is it optional?**

**mode:** either 'train' or 'test' depending on what we wish to do

**distribution:** the assumed distribution of the data

**formulas:** a dictionary with a list of formulas for each parameters of the distribution, see [Examples of training features](#Examples of training features) for more examples

**deep_models_dict:** a dictionary where keys are names of deep models and values are also dictionaries. In turn, their keys are 'model' with values been the model arcitectures and 'output_shape' with values been the output size of the model. Again see [Examples of training features](#Examples of training features) for examples.

**train_parameters:** A dictionary where the training parameters are defined. There are: batch size, epochs, optimizer, optimizer parameters and degrees of freedom of each parameter

### Data

There are two parameters required regarding the data, namely data and target. For both parameters two options are available:
* The user can give a local path which corresponds to a csv files storing the data. The SDDRDataset class will then load the data.
* The user has already loaded the data manually and sets data and target to two Pandas Data Frames corresponding to the input data and targert data.

A combination of the above options is also possible, i.e. have data as a dataframe and load target from a file and vice versa. 

### Distributions

Currently, the available distribution in PySDDR are:

* Normal: bernoulli distribution with logits (identity)
* Poisson: poisson with rate (exp)
* Bernoulli: bernoulli distribution with logits (identity)
* Bernoulli_prob: bernoulli distribution with probabilities (sigmoid)
* Multinomial: multinomial distribution parameterized by total_count(=1) and logits **-->is it implemented?**
* Multinomial_prob: multinomial distribution parameterized by total_count(=1) and probs
* Logistic: multinomial distribution parameterized by loc and scale

Note that when setting the distribution parameter the distribution names should be given exactly as above, as well as their parameters (which are required when defining formulas and degrees of freedom of each parameter).


### Deep Neural Networks

The neural network or networks to be used in the SDDR package are defined by the user. These user gives a name for each and this will be the corresponding key of the deep_models_dict input. Each value of the dictionary is of itself a dictionary with keys: model, where the neural network architecture is defined, and output_shape where the user should specify the output size of the neural network. This will help build the SDDR_Param_Net. The architecture can be either directly given, defined in a local script, or a pytorch model can be used.
 
#### Examples

1. Define two neural networks directly:
```
deep_models_dict = {
        'd1': {
            'model': nn.Sequential(nn.Linear(1,64)), nn.ReLU(), nn.Linear(64,8))
            'output_shape': 8},
        'd2': {
            'model': nn.Sequential(nn.Linear(1,32), nn.ReLU(), nn.Linear(32,4)),
            'output_shape': 4}
        }
```
Note that the correct imports will also need to be specified in your script, so for this case ``` import torch.nn as nn```

2. Use a model architecture saved in a script to define a network:
```
deep_models_dict = {
        'd1': {
            'model': myCNN(n_channels=1, n_outputs=8),
            'output_shape': 8},
        'd2': {
            'model': nn.Sequential(nn.Linear(1,32), nn.ReLU(), nn.Linear(32,4)),
            'output_shape': 4}
        }
```
Note that also here the correct import needs to be given, e.g. ``` from model import myCNN```

3. Use a pytorch model:
_This case we still need to test but i think it will work out of the box_
```
deep_models_dict = {
        'd0': {
            'model': models.resnet18(),
            'output_shape': 1}
        }
```

Note that also here the correct import needs to be given, e.g. ```import torchvision.models as models```

## Features

### Scientific features

_Here discuss the scientific features available to the user, e.g. w or w/o orthogonolization, optimizer options etc._

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
