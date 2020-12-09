# PySDDR

## What is this?

PySDDR is a python package used for regression tasks, which combines statistical regression models and neural networks into a general framework to deal with multi-modal data (e.g. tabular and image data). It can be used for mean regression as well as for distributional regression, i.e. estimating any parameter of the assumed distribution, not just the mean. Each distributional parameter is defined by a formula, consisting of a structured (statistical regression model) and unstructured (neural networks) part. One of the main advantages of this package is the introduction of an orthogonalization layer, which ensure identifiability when structured and unstructured parts share some input data, thus making the attribution of shared effects to either one of the parts (structured or unstructured) identifiable.

PySDDR allows beginners to easily use and take advantage of this general framework but also enables the more advanced user to exploit features and tweak parameters in a flexible and interactive way. The framework is written in PyTorch and accepts any number of neural networks, from the simplest classifier to the more complicated architectures, such as LSTMs. 

The python package was built based on the concepts presented in _paper_ and follows the R implementation found here _link_. 



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
This installation was tested for python version 3.7 and 3.8

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
    2.1. [User inputs](#User-inputs)  
    2.2. [Initialization](#Initialization)  
    2.3. [Training](#Training)  
    2.4. [Resume Training](#Resume-Training)  
    2.5. [Evaluating](#Evaluating)  
    2.6. [Saving](#Saving)  
    2.7. [Predicting](#Predicting)  
     




## Model


### SddrNet

The framework combines statistical regression models and neural networks into one larger unifying network - ```SddrNet``` - which has a dynamic network architecture because its architecture depends on the user input, i.e. assumed model distribution and defined formulas of distributional parameters. If ```SddrNet``` is used to build a distributional regression model, the user has to define a formula for each distributional parameter (e.g. a normal distribution has two parameters, *loc* and *scale*), which is then used by ```SddrNet``` to build a sub-network - ```SddrFormulaNet``` - for each distributional parameter. The output of each ```SddrFormulaNet``` is the predicted parameter value, which are collected by ```SddrNet```, transformed based on the distrubution's rules (e.g., an `exp` transformation to get a positive variance value) and then given as input to a distributional layer. From the distributional layer a regularized log loss is computed, which is then backpropagated. An example of this can be seen below.

![image](https://github.com/davidruegamer/PySDDR/blob/dev/images/sddr_net.jpeg)



### Preprocessing

Each distributional parameter is defined by a formula that consists of a structured (*blue*) and unstructured (*red*) part. 

![image](https://github.com/davidruegamer/PySDDR/blob/dev/images/formula.png)

The structured part can have linear (*lightblue*) and smoothing (*darkblue*, non-linear) terms, while the unstructured part consist of one or more neural network (*red*) terms. The user needs to define the input data for each term in the formula (the same input data can be assigned to different terms). While the structured part only accepts structured (tabular) data as input, the unstructured part accepts both, structured (tabular) and unstructured (currently only images are supported) data as input. During the preprocessing, the input data is assigned to the corresponding terms and for each smoothing term, the respective basis functions and penalty matrices are computed. The framework currently supports B-splines (default) and cyclic cubic regression splines. In a last step, the orthogonalization of the smoothing terms wrt. the linear terms is computed. The output of the processed structured part (linear and smoothing terms) is called structured features (consitsting of linear and smoothing features), while the output of the processed unstructured part (input to the neural networks) is called unstructured features.

### SddrFormulaNet

As mentioned, each ```SddrFormulaNet``` predicts a distributional parameter, based on the corresponding user-defined formula. The inputs to the ```SddrFormulaNet``` network are the processed structured and unstructured features. The structured features are concatenated and given to a fully connected layer, which we name Structured Head. The unstructured features are fed into one or multiple neural networks. Both the number and architecture of these networks are pre-defined by the user and are built within the ```SddrFormulaNet``` in a parallel fashion. Their outputs are concatenated and are given, together with the structured features, to the orthogonalization layer. Next, the orthogonalized, concatenated output of the neural networks is fed into a fully connected layer, which we name Deep Head. The sum of this output and the output of the Structured Head forms the parameter prediction of the SddrFormulaNet. An example of the architecture can be seen below.

![image](https://github.com/davidruegamer/PySDDR/blob/dev/images/sddr_formula_net.jpeg)

### Orthogonalization

Orthogonalization ensures identifiability of the input data by a decomposition of shared effects corresponding to the structured and unstructured part. It occurs in two parts of the network. The first orthogonalization is computed during preprocessing but only if linear features are a subset of the input of the smoothing terms. For example in ```~ 1 + x3 + spline(x3, bs='bs', df=9, degree=3)```, ```spline(x3, bs='bs', df=9, degree=3)``` is orthogonalized with respect to the intercept and x3. If any terms x2, x4 etc. are present they are ignored in this orthogonalization step. The second orthogonalization occurs in every forward step of the network and follows the same principle as before: it only occurs if linear or smoothing features are a subset of the input of the unstructured terms. For detailed description of the orthogonalization see _paper_ .


### Smoothing Penalty

For each spline in the formula, the number of basis function (```df```) and the degree of the spline functions (```degree```) have to be specified. A smoothing penalty matrix is implicity created when smoothing terms are used in the formula. Each smoothing penalty matrix is regularized with a lambda parameter computed from user-defined degrees of freedom. The degrees of freedom can be given as a single value, then all individual penalty matrices are multiplied with a single lambda. This ensures that no smoothing term has more flexibility than the other, which makes sense in certain situations. The degrees of freedom can also be given as a list, which not only allows to specify different degrees of freedom for each distributional parameter, but also to specify different degrees of freedom for each smoothing term in each formula by providing a vector of the same length as the number of smoothing terms in the parameter’s formula. In this case, all smoothing penalty matrices are multiplied by different lambdas.


## Sddr User Interface
 
The user interacts with the package through the Sddr class. An overview of this class and its iteraction with the rest of the package can be seen in the figure below:

![image](https://github.com/davidruegamer/PySDDR/blob/dev/images/sddr_user_interface.jpeg)

### User inputs 

There are a number of inputs which need to be defined by the user before training or testing can be performed. The user can either directly define these in their run script or give all the parameters through a config file - for more information see [Training features](#Training-features). 


#### Data

The data is required as input to three functions: ```sddr.train(), sddr.predict(), sddr.load()```.

**structured_data:** the structured data in tabular format. This data can either be pre-loaded into a pandas dictionary, which is then given as an input to the sddr functions, or the user can define a path to a csv file where the data is stored. Examples of these two options can be found in the beginner's guide tutorial.

**unstructured_data:** the unstructured data. Currently the PySDDR package only accepts images as unstructured data. The data is loaded in batches and for the initialization of the DataLoader a dictionary needs to be provided with information on the unstructured data. An example of this can be see below:

```
unstructured_data = {
    'x3':{
        'path': './mnist_data/mnist_images',
        'datatype': 'image'
    }
} 
```

The keys of the dictionary are the input feature names, as defined in the given formulas. Each input feature then has a dictionary with two keys, 'path' and 'datatype', where the path to the data directory is provided and the data type defined (currently only 'image' is accepted). 

**target:** the target data, i.e. our ground truth. This can be given with the same two options as above. Note that if the structured data has been given as a string or a pandas dataframe then the target data must also be given in the same format. However, structured data and target can be given in one dataframe, in which case the column name of the target with the dataframe needs to be provided as target.

A combination of the above options is also possible, i.e. have the structured data as a dataframe and load the target from a file and vice versa. Examples of these options can again be found in the beginner's guide tutorial.

#### Distributions

Currently, the available distribution in PySDDR are:

* Normal: normal distribution with mean(loc) and variance(scale)
* Poisson: poisson with rate(rate)
* Bernoulli: bernoulli distribution modeling the logits(logits)
* Bernoulli_prob: bernoulli distribution modeling probabilities(probs)
* Multinomial: multinomial distribution parameterized by total_count (default=1) and logits(logits)
* Multinomial_prob: multinomial distribution parameterized by total_count (default=1) and probabilities(probs)
* Logistic: logistic distribution parameterized by loc(loc) and scale(scale)

Note that when setting the ```distribution``` parameter, the distribution name should be given as above in string format, as well as their parameters (which are required when defining formulas and degrees of freedom of each parameter), e.g. ```distribution='Poisson'```.

#### Formulas

The PySDDR package uses Patsy in the backend to parse the formulas. Each formula is given as a string and follows Patsy's formula conventions described [here](https://patsy.readthedocs.io/en/v0.1.0/formulas.html).

An example of formulas for a Logistic distribution is given below:

```
formulas = {'loc': '~ 1 + x1 + spline(x1, bs="bs", df=4) + spline(x2, bs="bs",df=4) + d1(x3)+d2(x5)',
            'scale': '~ -1 + spline(x3, bs="bs",df=4) + spline(x4, bs="bs",df=4)'
            }
```

In this example, ```formulas``` has two keys, 'loc' and 'scale', corresponding to the two parameters of the Logistic distribution. The features x1, x2, x3, x4 need to be structured data as they are given as input to the linear and smoothing terms (structured part). This means that the tabular data needs to have column names corresponding to the features, i.e. x1, x2, x3, x4. Feature x5 can be either structured or unstructured data - note that if it is unstructured data 'x5' should be a key value in the unstructured_data input. In addition, an intercept is implicitely added to the formula but can also be explicitely added with an ```1``` (see ```loc```in fumula example above). If the intercept should be removed, a ```-1```statement can be added to the formula (see ```scale```in fumula example above). 


#### Deep Neural Networks

The neural networks to be used in the PySDDR package are defined by the user. The user provides a name for each neural network, which will be the corresponding name in the ```formulas``` and the corresponding key of the ```deep_models_dict```. Each value of the dictionary is itself a dictionary with keys: 'model', where the neural network architecture is defined, and 'output_shape', where the user should specify the output size of the neural network. This will help build the ```SddrFormulaNet```. The architecture can be either given directly, defined in a local script, or a pytorch model can be used.
 
**Examples**

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
Note that the correct imports will also need to be specified in your script, i.e. ``` import torch.nn as nn```

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


#### Train Parameters

The training parameters are: batch size, epochs, optimizer, optimizer parameters and degrees of freedom of each parameter. Batch size, epochs and degrees of freedom are required but defining the optimizer is optional. If no optimizer is defined by the user, *Adam* is used per default with PyTorch's default optimizer parameters, which can be found [here](https://pytorch.org/docs/stable/optim.html). An example of training parameters can be seen below:


```
 train_parameters = {
 'batch_size': 200,
 'epochs': 200,
 'optimizer': optim.SGD,
 'optimizer_params':{'lr': 0.01, 'momentum': 0.9}, 
 'degrees_of_freedom': {'loc':4, 'scale':4}
 }
 ```

Note that ```train_parameters['degrees_of_freedom']``` is a dictionary where the degrees of freedom of each parameter is defined. This can either be a list of degrees of freedom for each smoothing term in the formula or a single value for all smoothing terms.


### Initialization

A list of all required inputs during initialization of the Sddr instance can be seen next:

* distribution: the assumed distribution of the data, see more in [Distributions](#Distributions)
* formulas: a dictionary with a list of formulas for each parameters of the distribution, see more in [Formulas](#Formulas)
* deep_models_dict: a dictionary, where keys are names of deep models and values are also dictionaries. In turn, their keys are 'model' with values being the model arcitectures and 'output_shape' with values being the output size of the model. Again see [Deep Neural Networks](#Deep-Neural-Networks) for more details
* train_parameters: a dictionary, where the training parameters are defined, see more in [Train Parameters](#Train-Parameters)
* output_dir (optional): the path of the output directory (to save results)


*Example for initialization parameters:*

```
structured_data: ./X.csv

unstructured_data: {
  x3: {
    path: './images',
    datatype: 'image'
  }
} 

target: ./Y.csv

output_dir: ./outputs

mode: train
load_model: 

distribution: Poisson
formulas: {rate: '~ 1 + spline(x1, bs="bs",df=9) + spline(x2, bs="bs",df=9) + d1(x3) + d2(x2)'}

deep_models_dict: {
  d1: {
    model: 'models.alexnet()',
    output_shape: 1000},
  d2: {
    model: 'nn.Sequential(nn.Linear(1,3),nn.ReLU(), nn.Linear(3,8))',
    output_shape: 8}
}

train_parameters: {
  batch_size: 1000,
  epochs: 1000,
  optimizer: 'optim.SGD',
  optimizer_params: {lr: 0.01, momentum: 0.9}, 
  degrees_of_freedom: {rate: 10}
}

```

The initialization parameters have to be given to the Sddr instance: 

```
sddr = SDDR(data=data,
            target=target,
            output_dir=output_dir,
            distribution=distribution,
            formulas=formulas,
            deep_models_dict=deep_models_dict,
            train_parameters=train_parameters)
```

The initialization parameters can also be given as a ```config.yaml``` file and an Sddr instance can be initialized with:

```sddr = Sddr(config=config)```



### Training

For training two simple steps are required by the user. The following options are available:

 * Train with the structured training data: ```sddr.train(target, structured_data)```  
 * Train with the structured and unstructured training data: ```sddr.train(target, structured_data, unstructured_data)```  
 * Train, plot and save loss curve: ```sddr.train(target, structured_data, unstructured_data, plot=True)```  


### Resume Training

The user may also wish to load a pretrained model to resume training. This can be done by:

* Initialize an Sddr instance: ```sddr = Sddr(config=config)```
* Load the pre-trained model: ```sddr.load(model_name, training_data) ```
* Resume training: ```sddr.train(target, structured_data, resume=True)```


### Evaluating

The user can then evaluate the training on any distributional parameter, e.g. for a Poisson distribution: ```sddr.eval('rate') ```. This will return and plot the partial effects of the structured features. To turn off the plot functionality the user must set ```plot=False ``` when calling ```sddr.eval()```.

* To get the trained distribution the user can call ```distribution_layer = sddr.get_distribution()```. From this the user can then get all the properties available from [PyTorch's Probability Distributions package](https://pytorch.org/docs/stable/distributions.html) (torch.distributions), e.g. the mean can be retrieved by ```distribution_layer.mean``` or the standard deviaton by ```distribution_layer.stddev```.

* To get the trained network's weights, i.e. coefficients, for the structured part and for a specific distributional parameter, the user can call:```sddr.coeff('rate')```.


### Predicting

The trained model can be used to make predictions on unseen data. The following options are available:

* Predict on unseen structured data: ```sddr.predict(data) ```
* Predict on unseen structured and unstructured data: ```sddr.predict(data, unstructured_data) ```
* Predict on unseen data and clip the data range to fall inside training data range if a out of range error occurs: ```sddr.predict(data, clipping=True) ```
* Predict on unseed data and plot a figure for each spline defined in each formula of the distribution's parameters: ```sddr.predict(data, plot=True) ```

The distrubution as well as the partial effects for all structured features of all parameters of the distribution will be returned. 

Another option is to load a pre-trained model and use the previously trained model to make predictions on unseen data. This can be done by:

* Initialize an Sddr instance: ```sddr = Sddr(config=config) ```
* Load model: ```sddr.load(model_name, training_data) ``` 
* Predict on unseen data: ```sddr.predict(data) ```

Note here that the training data also needs to be provided during load for the preprocessing steps, i.e. basis functions creation, to be performed.


### Saving

After training and evaluation, the model can be saved by ```sddr.save()``` or ```sddr.save('MyModel.pth')```, if the user wishes to save the model with a name different than the default _model.pth_. This will be saved in the output directory defined by the user in the config, or if no output directory has been defined an _outputs_ directory is automatically created.




