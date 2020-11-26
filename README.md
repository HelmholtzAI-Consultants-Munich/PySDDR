# PySDDR

## What is this?

PySDDR is a python package used for regression tasks which combines statistical regression models and neural networks and takes advantage of the strengths of each to deal with multi-modal data.

It was built based on the concepts presented in _paper_ and follows the R implementation found here _link_. One of the main contributions of both the paper and packages is the introduction of an orthogonalization layer to help identifiability between structured and unstructured data.

The package works for both mean and distributional regression while assuming a distribution of any given data, as long as each parameter is defined by _a linear predictor_. It is built in such a way that allows for the beginner to easily use and take advantage of this unifying framework but also enables the more advanced user to exploit features and tweak parameters in a flexible and interactive way. The main model is built in a dynamic way depending on the user input and can accept any number of neural networks, from the simplest classifier to the more complicated architectures, such as LSTMs, integrating all these in a unified network written in PyTorch. Meanwhile the structured data is smoothed using splines as basis function and its partial effects can be visualized during evaluation.

## Installation

To install the package run:

```
pip install .        (Installation as python package: run inside directory)
```
or if you want to develop the package:
```
pip install -e .        (Installation as python package: run inside directory)
```

If you are using conda first install pip by: ```conda install pip```

## Tutorials

Two tutorials are available in the [tutorials](https://github.com/davidruegamer/PySDDR/tree/dev/tutorials) directory. One is aimed as a beginner's guide, where toy data is used to familiarize the user with the package and the second includes a more advanced tutorial with an example of applying the package also to unstructured data.

## Contents

1. [Model](#Model)
1.1. [SddrNet](#SddrNet)
1.2. [Preprocessing](#Preprocessing)
1.3. [SddrFormulaNet](#SddrFormulaNet)
1.4. [Orthogonalization](#Orthogonalization)
1.5. [Smoothing Penalty](#Smoothing-Penalty)
2. [User Interface](#Sddr-User-Interface)
2.1. [Training](#Training)
2.2. [Evaluating](#Evaluating)
2.3. [Saving](#Saving)
2.4. [Predicting](#Predicting)
2.5. [Resume Training](#Resume-Training)
2.6. [User inputs](#User-inputs)


## Model


### SddrNet

The model architecture is built dynamically, depending on the user inputs such as the assumed distribution and _linear predictors_. It combines statistical models and neural networks into one larger unifying network, namely SddrNet, responsible for integrating all parts. Depending on the number of parameters of the assumed distribution defined by the user, SDDRNet consists of a number of smaller networks, SddrFormulaNet, which are built in parallel. A formula is given by the user for each parameter based on which each SddrFormulaNet is built and the output of each is the predicted parameter value. Within SddrNet, these are collected, normalized based on the distrubution's rules, and given as input to a distributional layer. From the distributional layer a log loss is computed, to which a smoothing penalty is added for regularization forming the final loss which is then backpropagated. SddrNet accepts the data after preprocessing has been applied. An example of this can be seen below.

![image](https://github.com/davidruegamer/PySDDR/blob/dev/images/sddr_net.jpg)

### Preprocessing

PySDDR has been built to accept tabular and imaging data, both singly and combined. The individual features of these data can be of two types: structured and unstructured. The user needs to define which features belong to each of these types and this is done through the parameters' formulas. The first part of preprocessing is then to split the input data into structured and unstructured depending on the given formulas. The structured part can have linear and non-linear (smoothing splines) terms, while the unstructured part can consist of one or more neural networks. [The non-linear strcuctured terms smoothing terms]
Currently b-splines are used per default, whereas Cyclic Cubic splines are also available.

### SddrFormulaNet

As mentioned, each SddrFormulaNet predicts a parameter of the assumed distribution. Depending on the given formula the SddrFormulaNet is built. The inputs to the network are the processed structured data and the unstructured data. The processed structured data (linear and non-linear terms) is concatenated and given to a fully connected layer, which we name Structured Head. The unstructured data is given into one or multiple neural networks. Both the number and architecture of these are pre-defined by the user and are built within the SddrFormulaNet in a parallel fashion. Their outputs are concatenated and together with the processed structured data are given to the orthogonalization layer. Next, the orthogonalized, concatenated output of the neural networks is fed into a fully connected layer, which we name Deep Head. The sum of this output and the output of the Structured Head forms the parameter prediction of the SddrFormulaNet. An example of the architecture can be seen below.

![image](https://github.com/davidruegamer/PySDDR/blob/dev/images/sddr_param_net.jpg)

### Orthogonalization

Orthogonalization ensures identifiability of the data by a decomposition of covariates corresponding to the linear, structured and unstructured  part of the data.
It occurs in two parts of the network. The first is performed once during preprocessing and only if linear features are a subset of the structured inputs. For example ```spline(x3, bs='bs', df=9, degree=3)``` is orthogonalized with respect to the intercept and x3. If any terms x2, x4 etc. are present they are ignored in this orthogonalization step. The second orthogonalization occurs in every forward step of the network and follows the same principle as before: it only occurs if structured features are a subset of the unstructured inputs. The formula used for the ortogonalization is the same in both cases and can be described as follows:

Assume we have structured data $$X$$ and unstructured data $$U$$ which pass through the deep networks (defined by the user) and concatenated giving latent features $$\hat{U} = d(U)$$. Then we can replace $$\hat{U}$$ with $$\tilde{U} = P_{orthog}\hat{U}$$

For structured head weights $$w$$ and deep head weights $$\gamma$$ the ouptut of the SddrFormulaNet will then be:

$$ \eta = Xw + \tilde{U}\gamma$$

### Smoothing Penalty







## Sddr User Interface
 
The user interacts with the package through the Sddr class. An overview of this class and its iteraction with the rest of the package can be seen in the fgigure below:

[LISAS FIG]


### Training

For training two simple steps are required by the user:

* Initialize an Sddr instance, e.g. ```sddr = Sddr(config=config)```
* Train with the structured training data by ```sddr.train(target, structured_data)```
* Train with the structured and unstructured training data by ```sddr.train(target, structured_data, unstructured_data)```
* Train, plot and save loss curve by ```sddr.train(target, structured_data, unstructured_data, plot=True)```

### Evaluating

The user can then perform an evalutation of the training on any of the parameter's distribution, e.g. for a Poisson distribution: ```sddr.eval('rate') ```. This will return and plot the partial effects of the structured features. To turn off the plot functionality the user must set ```plot=False ``` when calling ```sddr.eval```.

* To get the trained distribution the user can call ```distribution_layer = sddr.get_distribution()```. From this the user can then get all the properties avalaible from [PyTorch's Probability Distributions package](https://pytorch.org/docs/stable/distributions.html) (torch.distributions), e.g. the mean can be retrieved by ```distribution_layer.mean``` or the standard deviaton by ```distribution_layer.stddev```.

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

### Resume Training

The user may also wish to load a pretrained model to resume training. For this the first two steps (init, load) from above need to be performed and then the user can resume training by ```sddr.train(target, structured_data, resume=True)```


### User inputs 

#### Sddr Initialization

There are a number of inputs which need to be defined by the user before training, or testing can be performed. The user can either directly define these in their run script or give all the parameters through a config file - for more information see [Training features](#Training features). 

A list of all required inputs during initialization of the sddr instance can be seen next:

**distribution:** the assumed distribution of the data, see more in [Distributions](#Distributions)

**formulas:** a dictionary with a list of formulas for each parameters of the distribution, see more in [Formulas](#Formulas)

**deep_models_dict:** a dictionary where keys are names of deep models and values are also dictionaries. In turn, their keys are 'model' with values being the model arcitectures and 'output_shape' with values being the output size of the model. Again see [Deep Neural Networks](#Deep-Neural-Networks) for more details

**train_parameters:** A dictionary where the training parameters are defined, see more in [Train Parameters](#Train-Parameters)

Additionally, the path of the output directory in which to save results  can be defined by the user by setting: **output_dir** 

#### Data

The data is required as input to three functions: ```sddr.train, sddr.predict, sddr.load```.

**structured_data:** the structured data in tabular form. This data can either be pre-loaded into a pandas dictionary which is then given as an input to the sddr functions, or the user can define a path to a csv file where the data is stored. Examples of these two options can be found in the beginner's guide tutorial.

**unstructured_data:** the unstructured data. Currently the PySDDR package only expects images as unstructured data (but we aim to extend this soon to include text). The data is loaded in batches and for the initialization of the DataLoader a dictionary needs to be provided with information on the unstructured data. An example of this can be see below:

```
unstructured_data = {
    'x3':{
        'path': './mnist_data/mnist_images',
        'datatype': 'image'
    }
} 
```

The keys of the dictionary are the feature names as defined also in the given formulas. Each feature then has a dictionary with two keys, 'path' and 'datatype', where the path to the data directory is provided and the data type defined (currently only 'image' is accepted). 

**target:** the target data, i.e. our ground truth. This can also be given witht he same two options as above. Note that if the structured data has been given as a string or a pandas dataframe then the target data must also be given in the same format. However, one dataframe can be given for both in which case the column name of the target with the dataframe needs to be provided as target.

A combination of the above options is also possible, i.e. have data as a dataframe and load target from a file and vice versa. Examples of these options can again be found in the beginner's guide tutorial.

#### Distributions

Currently, the available distribution in PySDDR are:

* Normal: bernoulli distribution with logits (identity)
* Poisson: poisson with rate (exp)
* Bernoulli: bernoulli distribution with logits (identity)
* Bernoulli_prob: bernoulli distribution with probabilities (sigmoid)
* Multinomial: multinomial distribution parameterized by total_count(=1) and logits **-->is it implemented?**
* Multinomial_prob: multinomial distribution parameterized by total_count(=1) and probs
* Logistic: multinomial distribution parameterized by loc and scale

Note that when setting the ```distribution``` parameter the distribution name should be given exactly as above in sting format, as well as their parameters (which are required when defining formulas and degrees of freedom of each parameter), e.g. ```distribution='Poisson'```

#### Formulas

The PySDDR package uses Patsy in the backend to parse the formulas. Each formula is given as a string and follows Patsy's formula conventions described [here](https://patsy.readthedocs.io/en/v0.1.0/formulas.html).

An example of formulas for a Logistic distribution is given below:

```
formulas = {'loc': '~1+x1+spline(x1, bs="bs", df=4)+spline(x2, bs="bs",df=4) + d1(x3)+d2(x5)',
            'scale': '~1 + spline(x3, bs="bs",df=4) + spline(x4, bs="bs",df=4)'
            }
```
Formulas here has two keys loc and scale, corresponding to the two parameters of the Logistic distribution. The features x1,x2,x3,x4 need to be structured data as they are given as input both to the linear term and to the splines (structured part). This means that the tabular data needs to have column names corresponding to the features, i.e. x1,x2,x3,x4. Feature x5 can be either structured or unstructured - note that if it is unstructured 'x5' should be a key value in the unstructured_data input. 


### Deep Neural Networks

The neural network or networks to be used in the SDDR package are defined by the user. These user gives a name for each and this will be the corresponding key of the deep_models_dict input. This also then needs to be given in the same way in the formulas. Each value of the dictionary is of itself a dictionary with keys: model, where the neural network architecture is defined, and output_shape where the user should specify the output size of the neural network. This will help build the SddrFormulaNet. The architecture can be either directly given, defined in a local script, or a pytorch model can be used.
 
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
```
deep_models_dict = {
        'd0': {
            'model': models.resnet18(),
            'output_shape': 1}
        }
```

Note that also here the correct import needs to be given, e.g. ```import torchvision.models as models```

The last two methods can only be used if the class inputs are defined in a python script, they are currently not available when loading the inputs from a config file.


### Train Parametes

The training parameters are: batch size, epochs, optimizer, optimizer parameters and degrees of freedom of each parameter. Batch size, epochs and degrees of freedom are required but defining the optimizer is optional. If no optimizer is defined by the user Adam is used per default with PyTorch's default optimizer parameters, which can be found [here](https://pytorch.org/docs/stable/optim.html). An example of training parameters can be seen below:


```
 train_parameters = {
 'batch_size': 200,
 'epochs': 200,
 'optimizer': optim.SGD,
 'optimizer_params':{'lr': 0.01, 'momentum': 0.9}, 
 'degrees_of_freedom': {'loc':4, 'scale':4}
 }
 ```

Note that ```train_parameters['degrees_of_freedom']``` is a dictionary where the degrees of freedom of each parameter is defined. This can either be a list of degrees of freedom for each spline in the formula or a single number (same degrees of freedom for all splines). Using the Demmler-Reinsch Orhtogonalization, all smoothing terms are then calculated based on this specification (e.g., setting degrees_of_freedom = 5 results in sp = 1.234 for one smooth, but sp = 133.7 for another smooth due to their different nature and data). This ensures that no smooth term has more flexibility than the other term which makes sense in certain situations.
