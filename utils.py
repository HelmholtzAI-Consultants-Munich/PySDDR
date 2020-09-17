import numpy as np
from patsy import dmatrix
import statsmodels.api as sm
import pandas as pd
import os
from torch import nn

from statsmodels.gam.api import CyclicCubicSplines, BSplines

def checkups(family, formulas, cur_distribution):
    """
    Checks if the user has given an available distribution, too many formulas or wrong parameters for the given distribution
    Parameters
    ----------
        family : dictionary
            A dictionary holding all the available distributions as keys and values are again dictionaries with the parameters as keys and 
            values the formula which applies for each parameter 

        formulas : dictionary
            A dictionary with keys corresponding to the parameters of the distribution defined by the user and values to strings defining the
            formula for each distribution, e.g. formulas['loc'] = '~ 1 + bs(x1, df=9) + dm1(x2, df=9)'
            
        cur_distribution : string
            The current distribution defined by the user
            
    Returns
    -------
        new_formulas : dictionary
            If the current distribution is available in the list of families, new_formulas holds a formula for each parameter of the distribution.
            If a formula hasn't been given for a parameter and ~0 formula is set. If the current distribution is not available an empty dictionary
            is returned.  
    """
    # return an empty dict if distribution not available
    if cur_distribution not in family.keys():
        print('Distribution not in family of distributions! Available distributions are: ', list(family.keys()))
        return dict() 
    else:
        #if len(formulas) > len(family[cur_distribution]):
        # check either if too many formulas have been given and drop them or if wrong parameter names have been given
        new_formulas=dict()
        for param in family[cur_distribution].keys():
            if param in formulas:
                new_formulas[param] = formulas[param]
            # define an empty formula for parameters for which the user has not specified a formula
            else:
                print('Parameter formula', param,'for distribution not defined. Creating a zero formula for it.')
                new_formulas[param] = '~0'
        return new_formulas

def split_formula(formula, net_names_list):
    """
    Splits the formula into two parts - the structured and unstructured part
    Parameters
    ----------
        formula : string
            The formula to be split, e.g. '~ 1 + bs(x1, df=9) + dm1(x2, df=9)'
            
        net_names_list : list of strings
            A list of all newtwork names defined by the user
            
    Returns
    -------
        structured_part : string
            A string holding only the structured part of the original formula
        unstructured_terms: list of strings
            A list holding all the unstructured parts of the original formula   
    """
    structured_terms = []
    unstructured_terms = []
    # remove spaces the tilde and split into formula terms
    formula = formula.replace(' ','')
    formula = formula.replace('~','')
    formula_parts = formula.split('+')
    # for each formula term
    for part in formula_parts:
        term = part.split('(')[0]
        # if it an unstructured part
        if term in net_names_list:
            # append it to a list
            unstructured_terms.append(part)
        else:
            structured_terms.append(part)
    # join the structured terms together again
    structured_part = '+'.join(structured_terms)    
    return structured_part, unstructured_terms

def spline(x,bs="bs",df=4, degree=3,return_penalty = False):
    if bs == "bs":
        s = BSplines(x, df=[df], degree=[degree], include_intercept=True)
    elif bs == "cc":
        s = CyclicCubicSplines(x, df=[df])
    else:
        print("Spline basis not defined!")

    #return either basis or penalty
    if return_penalty:
        return s.penalty_matrices
    else:
        return s.basis

def _get_P_from_design_matrix(dm, data):

    factor_infos = dm.design_info.factor_infos
    terms = dm.design_info.terms
    
    big_P = np.zeros((dm.shape[1],dm.shape[1]))
    
    column_counter = 0
    
    for term in terms:
        if len(term.factors) == 0:
            column_counter += 1
        else:
            factor_info = factor_infos[term.factors[0]]
            factor = factor_info.factor
            state = factor_info.state.copy()
            num_columns = factor_info.num_columns

            
            #here the hack starts
            code = state['eval_code']

            is_spline = code.split("(")[0] == "spline"
            if is_spline:
                code.replace('return_penalty = False','')
                state['eval_code'] = code[:-1] + ", return_penalty = True)"

                P = factor.eval(state, data)
                big_P[column_counter:(column_counter+num_columns),column_counter:(column_counter+num_columns)] = P[0]     
            column_counter += num_columns

    return big_P


def parse_formulas(family, formulas, data, cur_distribution, deep_models_dict, deep_shapes):
    """
    Parses the formulas defined by the user and returns a dict of dicts which can be fed into SDDR network

    Parameters
    ----------
        family : dictionary
            A dictionary holding all the available distributions as keys and values are again dictionaries with the parameters as keys and 
            values the formula which applies for each parameter 
        formulas : dictionary
            A dictionary with keys corresponding to the parameters of the distribution defined by the user and values to strings defining the
            formula for each distribution, e.g. formulas['loc'] = '~ 1 + bs(x1, df=9) + dm1(x2, df=9)'
        data: Pandas.DataFrame
            A data frame holding all the data 
        cur_distribution : string
            The current distribution defined by the user
        deep_models_dict: dictionary 
            A dictionary where keys are model names and values are instances
        deep_shapes: dictionary
            A dictionary where keys are network names and values are the number of output features of the networks
    Returns
    -------
        parsed_formula_contents: dictionary
            A dictionary where keys are the distribution's parameter names and values are dicts. The keys of these dicts will be: 'struct_shapes',
            'P', 'deep_models_dict' and 'deep_shapes'
        meta_datadict: dictionary
            A dictionary where keys are the distribution's parameter names and values are dicts. The keys of these dicts will be: 'structured' and 
            neural network names if defined in the formula of the parameter (e.g. 'dm1'). Their values are the data for the structured part (after 
            smoothing for the non-linear terms) and unstructured part(s) of the SDDR model 
    """
    # perform checks on given distribution name, parameter names and number of formulas given
    formulas = checkups(family, formulas, cur_distribution)
    if not formulas:
        exit
    meta_datadict = dict()
    parsed_formula_contents = dict()
    struct_list = []
    # for each parameter of the distribution
    for param in formulas.keys():
        meta_datadict[param] = dict()
        parsed_formula_contents[param] = dict()
        structured_part, unstructured_terms = split_formula(formulas[param], list(deep_models_dict[param].keys()))
        print('results from split formula')
        print(structured_part)
        print(unstructured_terms)
        if not structured_part:
            structured_part='~0'
        
        structured_matrix = dmatrix(structured_part, data, return_type='dataframe')
        P = _get_P_from_design_matrix(structured_matrix, data)

        meta_datadict[param]['structured'] = structured_matrix.values
        parsed_formula_contents[param]['struct_shapes'] = structured_matrix.shape[1]
        parsed_formula_contents[param]['P'] = P
        parsed_formula_contents[param]['deep_models_dict'] = dict()
        parsed_formula_contents[param]['deep_shapes'] = dict()
        if unstructured_terms:
            for term in unstructured_terms:
                term_split = term.split('(')
                net_name = term_split[0]
                feature_names = term_split[1].split(')')[0]
                feature_names_list = feature_names.split(',')
                unstructured_data = data[feature_names_list]
                unstructured_data = unstructured_data.to_numpy()
                meta_datadict[param][net_name] = unstructured_data
                parsed_formula_contents[param]['deep_models_dict'][net_name]= deep_models_dict[param][net_name]
                parsed_formula_contents[param]['deep_shapes'][net_name] = deep_shapes[param][net_name] #Dominik: can we not just say unstructured_data.shape[1] here? so the user does not have to provide this shapes?
                # Christina: this is the shape of the output of the deep models not the shape of the input, e.g. if you have a nn.Linear(1,5) the deep_shape=5 whereas unstructured_data.shape[1]=1

    return parsed_formula_contents, meta_datadict


class create_family():
    
    def __init__(self):        
        self.families = {'Normal':{'loc': 'whateva', 'scale': 'whateva2'}, 
                       'Poisson': {'rate': 'whateva'}, 
                       'Gamma':{'concentration': 'whateva', 'rate': 'whateva'},
                       'Beta':{'concentration1': 'whateva', 'concentration0': 'whateva'},
                       'Bernoulli':{'logits': 'whateva'},
                       'Bernoulli_prob':{'probs':'whateva'},
                       'Multinomial':{'probs':'whateva'},
                       'NegativeBinomial':{'loc': 'whateva', 'scale': 'whateva2'}}
        
    def get_distribution_layer_type(self, family):   
        
        if family == "Normal":
            distribution_layer_type = torch.distributions.normal.Normal
        elif family == "Poisson":
            distribution_layer_type = torch.distributions.poisson.Poisson
        elif family == "Gamma":
            distribution_layer_type = torch.distributions.gamma.Gamma
        elif family == "Beta":
            distribution_layer_type = torch.distributions.beta.Beta
        elif family == "Bernoulli" or self.family == "Bernoulli_prob":
            distribution_layer_type = torch.distributions.bernoulli.Bernoulli       
        elif family == "Multinomial":
            distribution_layer_type = torch.distributions.multinomial.Multinomial 
        elif family == "NegativeBinomial":
            distribution_layer_type = torch.distributions.negative_binomial.NegativeBinomial
        else:
            raise ValueError('Unknown distribution')
           
        return distribution_layer_type
        
    def get_distribution_trafos(self, family, pred):
        pred_trafo = dict()
        add_const = 1e-8
        
        if family == "Normal":
            pred_trafo["loc"] = pred["loc"]
            pred_trafo["scale"] = add_const + pred["scale"].exp()
            
        elif family == "Poisson":
            pred_trafo["rate"] = add_const + pred["rate"].exp()
            
        elif family == "Gamma":
            pred_trafo["concentration"] = add_const + pred["concentration"].exp()
            pred_trafo["rate"] = add_const + pred["rate"].exp()
            
        elif family == "Beta":
            pred_trafo["concentration1"] = add_const + pred["concentration1"].exp()
            pred_trafo["concentration0"] = add_const + pred["concentration0"].exp()
            
        elif family == "Bernoulli":
            pred_trafo["logits"] = pred["logits"]
            
        elif family == "Bernoulli_prob":
            pred_trafo["probs"] = torch.nn.functional.sigmoid(pred["probs"])
            
        elif family == "Multinomial":
            pred_trafo["total_count"] = 1
            pred_trafo["probs"] = torch.nn.functional.softmax(pred["probs"])
            
        elif family == "NegativeBinomial":   
            ####### to do: loc, scale -> f(total count) , p(probs)
            pred_trafo["total_count"] = pred["total_count"]  # constant
            pred_trafo["probs"] = pred["probs"]
            
        else:
            raise ValueError('Unknown distribution')
                 
        return pred_trafo
    

if __name__ == '__main__':
    # test
    '''
    dir_path = r'/Users/christina.bukas/Documents/AI_projects/code/reimplementations/example_data/simple_gam/'
    x_path = os.path.join(dir_path, 'X.csv')
    x = pd.read_csv (x_path, sep=';',header=None)
    x = x.to_numpy()
    xx = pd.DataFrame({'x1':x[:,0], 'x2':x[:,1]})
    '''
    # load data
    iris = sm.datasets.get_rdataset('iris').data
    x = iris.rename(columns={'Sepal.Length':'x1','Sepal.Width':'x2','Petal.Length':'x3','Petal.Width':'x4','Species':'y'})
    # define formulas
    formulas = dict()
    formulas['loc'] = '~1+bs(x1, df=9)+d1(x1)+d2(x2)'
    deep_models_dict = dict()
    deep_models_dict['loc'] = dict()
    deep_models_dict['loc']['d1'] = nn.Sequential(nn.Linear(1,10))
    deep_models_dict['loc']['d2'] = nn.Sequential(nn.Linear(10,3),nn.ReLU(), nn.Linear(3,8))
    deep_shapes = dict()
    deep_shapes['loc'] = {'d1' : 10, 'd2' : 8}

    #formulas['scale'] = '~d2(x4,x1)'
    #formulas['n'] = '~9+sb(x1)'
    # define distributions and network names
    cur_distribution = 'poisson'
    family = {'normal':{'loc': 'whateva', 'scale': 'whateva2'}, 'poisson': {'loc': 'whateva'}, 'binomial':{'n': 'whateva', 'p': 'whateva'}}
    # geta meta_datadict
    meta_datadict = parse_formulas(family, formulas, x, cur_distribution, deep_models_dict, deep_shapes)
    #print(meta_datadict)

        
'''
#manual parser - UNDONE        
def parse_formula(formula, data, param, list_of_nn_names):
    
    # remove spaces from formula
    formula.replace(' ', '')
    # check if formula is also left-sided
    assert formula[0] != '~', 'Only one-sided formulas allowed.'
    # split formula
    formula_split = formula.split('+')
    # check if null model is given
    if formula == '~0'
        print('Null model given for parameter: ', param)
        return {}
    # or intercept only
    elif formula == '~1':
        return {'structured': np.ones(data.shape[1])}
    
    # go through parts of the formula
    for formula_part in formula_split:
        # regular expressions??
        # check if there is a (...) in formula_part
        # if yes check left part
            #if yes check if it is s or part of nn names
            # if no check if it is 0,1, or part of data names
            
        if formula_part[0] == 's': 
            # perform smoothing
            input_data_name = formula_part[0].split('(')[-1].split(')')[0]
            smooth_data = smoothing[data[input_data_name]]
            # extend later to get penalty function
            struct_list.append(smooth_data)
        # unstructured part    
        elif formula_part[0] == 'd':
            model_name = formula_part[0].split('(')[0]
            input_data = formula_part[0].split('(')[-1].split(')')[0]
            data_dict[model_name] = data[input_data_name]
        # linear part
        else:
            # check if wrong e.g.3
            l = ..
            # first row of matrix should be the linear part
            struct_list.append(l)
    # concatenate l,sx1,sx2 and put matrix into
    data_dict['structured'] = np.concatenate(struct_list, dim=1)
    return data_dict
'''


