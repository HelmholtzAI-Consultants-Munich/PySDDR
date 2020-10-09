import numpy as np
from patsy import dmatrix
import statsmodels.api as sm
import pandas as pd
import os
from torch import nn
import torch
import parser

from statsmodels.gam.api import CyclicCubicSplines, BSplines

def _checkups(params, formulas):
    """
    Checks if the user has given an available distribution, too many formulas or wrong parameters for the given distribution
    Parameters
    ----------
        params : list of strings
            A list of strings of the parameters of the current distribution

        formulas : dictionary
            A dictionary with keys corresponding to the parameters of the distribution defined by the user and values to strings defining the
            formula for each distribution, e.g. formulas['loc'] = '~ 1 + spline(x1, bs="bs", df=9) + dm1(x2)'
            
            
    Returns
    -------
        new_formulas : dictionary
            If the current distribution is available in the list of families, new_formulas holds a formula for each parameter of the distribution.
            If a formula hasn't been given for a parameter and ~0 formula is set. If the current distribution is not available an empty dictionary
            is returned.  
    """
    new_formulas=dict()
    for param in params:
        if param in formulas:
            new_formulas[param] = formulas[param]
        # define an empty formula for parameters for which the user has not specified a formula
        else:
            print('Parameter formula', param,'for distribution not defined. Creating a zero formula for it.')
            new_formulas[param] = '~0'
    return new_formulas

def _split_formula(formula, net_names_list):
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

def spline(x, bs="bs", df=4, degree=3, return_penalty = False):
    """
    Computes basis functions
    Parameters
    ----------
        x: Pandas.DataFrame
            A data frame holding all the data 
        bs: string, default is 'bs'
            The type of splines to use - default is b splines, but can alos use ccyclic cubic splines if bs='cc'
        df: int, default is 4
            Number of degrees of freedom (equals the number of columns in s.basis)
        degree: int, default is 3
            degree of polynomial e.g. 3 -> cubic, 2-> quadratic
        return_penalty: bool, default is False
            If False s.basis is returned, else s.penalty_matrices is returned
    Returns
    -------
        The function returns one of:
        s.basis: The basis functions
        s.penalty_matrices: The penalty matrices of the splines 
    """
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
    """
    Computes and returns the penalty matrix
    Parameters
    ----------
        dm: patsy.dmatrix
            The design matrix for the structured part of the formula - computed by patsy
        data: Pandas.DataFrame
            A data frame holding all the data 
    Returns
    -------
        big_P: numpy array
            The penalty matrix of the design matrix
    """
    factor_infos = dm.design_info.factor_infos
    terms = dm.design_info.terms
    
    big_P = np.zeros((dm.shape[1],dm.shape[1]))
    
    column_counter = 0
    
    for term in terms:
        if len(term.factors) != 1: #currently we only use smoothing for 1D, in the future we also want to add smoothing for tensorproducts
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
                code.replace(' ','')
                code.replace(',return_penalty = False','')

                state['eval_code'] = code[:-1] + ", return_penalty = True)"

                P = factor.eval(state, data)
                big_P[column_counter:(column_counter+num_columns),column_counter:(column_counter+num_columns)] = P[0]     
            column_counter += num_columns

    return big_P

def _get_input_features_for_functional_expression(functional_expression : str, feature_names : list):
    '''
    Parses variables from a functional expression using the python parser
    Parameters
    ----------
        functional_expression: string
            functional expression from which to extract the input features like "spline(x1,x2, bs="bs", df=4, degree=3)"
            
        feature_names: set
            set of all possible feature names in the data set like [x1,x2,x3,x4,x5]
            
    Returns
    -------
        input_features: list
            list of feature names that appear as input in functional_expression. here in the example ["x1","x2"]
    '''
    co_names = parser.expr(functional_expression).compile().co_names #co names are local variables of functions in a python expression
    co_names_set = set(co_names)
    input_features = list(co_names_set.intersection(set(feature_names)))
    return input_features


def _get_all_input_features_for_term(term, feature_names):
    '''
    Extracts all feature names that appear in a patsy term. For this it loops through all factors and uses then a python code paser to extract input variables.
    Parameters
    ----------
        term: patsy term object
            patsy term object for which the feature names should be extracted
            
        feature_names: set
            set of all possible feature names in the data set like [x1,x2,x3,x4,x5]
            
    Returns
    -------
        input_features_term: list
            list of feature names that appear in the patsy term. e.g. for a term x1:spline(x2, bs="bs", df=4, degree=3) -> ["x1","x2"]
    '''
    
    factors = term.factors
    input_features_term = set()
    for factor in factors:
        factor_name = factor.name()
        input_features_factor = _get_input_features_for_functional_expression(factor_name, list(feature_names))
        input_features_term.union(set(input_features_factor))
        
    input_features_term = list(input_features_term)
    return input_features_term

def _get_info_from_design_matrix(structured_matrix, feature_names):
    """
    Parses the formulas defined by the user and returns a dict of dicts which can be fed into SDDR network
    Parameters
    ----------
        structured_matrix: patsy.dmatrix
            The design matrix for the structured part of the formula - computed by patsy
    Returns
    -------
        has_intercept: boolean
            If intercept column returns True else returns False
        list_of_dfs: list of ints
            A list containing the degrees of freedom of each spline used to compute the design matrix
        list_of_features: list of strings
            A list of lists. Each item in the parent list corresponds to one spline (used to compute the design matrix) and 
            is a list of the names of the features sent as input into that spline.
    """
    list_of_dfs = []
    list_of_spline_input_features = []
    has_intercept = 'False'
    for term in structured_matrix.design_info.terms:
        dm_term_name = term.name()
        if dm_term_name == "Intercept":
            has_intercept = 'True'
        elif 'spline' in dm_term_name:
            # get the number of columns produced by the spline = number of dofs of the spline
            number_of_columns = np.prod([structured_matrix.design_info.factor_infos[factor].num_columns for factor in term.factors])
            # get the feature names sent as input to each spline
            # old code
            #split_term = dm_term_name.split('(')[-1]
            #split_term = split_term.split(', bs')[0]
            #feature_names = split_term.split(',')
            # Dominik: changed that code, ad the parsing was not robust to e.g. change of order of arguments for spline. e.g. spline(x, df=4, bs="bs", degree=3)
            feature_names_spline = _get_all_input_features_for_term(term, feature_names)

            
            # append to lists
            list_of_spline_input_features.append(feature_names_spline)
            list_of_dfs.append(number_of_columns)
    return has_intercept, list_of_dfs, list_of_spline_input_features

def parse_formulas(family, formulas, data, deep_models_dict, verbose=False):
    """
    Parses the formulas defined by the user and returns a dict of dicts which can be fed into SDDR network
    Parameters
    ----------
        family: dictionary
            A dictionary holding all the available distributions as keys and values are again dictionaries with the 
            parameters as keys and values the formula which applies for each parameter 
        formulas: dictionary
            A dictionary with keys corresponding to the parameters of the distribution defined by the user and values
            to strings defining the formula for each distribution, e.g. formulas['loc'] = '~ 1 + spline(x1, bs="bs", df=9) + dm1(x2)'
        data: Pandas.DataFrame
            A data frame holding all the data 
        cur_distribution : string
            The current distribution defined by the user
        deep_models_dict: dictionary 
            A dictionary where keys are model names and values are dicts with model architecture and output shapes
    Returns
    -------
        parsed_formula_contents: dictionary
            A dictionary where keys are the distribution's parameter names and values are dicts. The keys of these dicts 
            will be: 'struct_shapes', 'P', 'deep_models_dict' and 'deep_shapes' with corresponding values
        meta_datadict: dictionary
            A dictionary where keys are the distribution's parameter names and values are dicts. The keys of these dicts 
            will be: 'structured' and neural network names if defined in the formula of the parameter (e.g. 'dm1'). Their values 
            are the data for the structured part (after smoothing for the non-linear terms) and unstructured part(s) of the SDDR 
            model 
         dm_info_dict: dictionary
            A dictionary where keys are the distribution's parameter names and values are dicts containing: a bool of whether the
            formula has an intercept or not and a list of the degrees of freedom of the splines in the formula and a list of the inputs features for each spline
    """
    # perform checks on given distribution name, parameter names and number of formulas given
    formulas = _checkups(family.get_params(), formulas)
    meta_datadict = dict()
    parsed_formula_contents = dict()
    struct_list = []
    dm_info_dict = dict()
    
    # for each parameter of the distribution
    for param in formulas.keys():
        meta_datadict[param] = dict()
        parsed_formula_contents[param] = dict()
        
        # split the formula into sructured and unstructured parts
        structured_part, unstructured_terms = _split_formula(formulas[param], list(deep_models_dict.keys()))
        
        # print the results of the splitting if verbose is set
        if verbose:
            print('results from split formula')
            print(structured_part)
            print(unstructured_terms)
            
        # if there is not structured part create a null model?
        if not structured_part:
            structured_part='~0'
            
        # create the structured matrix from the structured part of the formula - based on patsy
        structured_matrix = dmatrix(structured_part, data, return_type='dataframe')
        
        # get bool depending on if formula has intercept or not and degrees of freedom and input feature names for each spline
        has_intercept, list_of_dfs, list_of_spline_input_features = _get_info_from_design_matrix(structured_matrix, feature_names = data.columns)
        dm_info_dict[param] = {'has_intercept': has_intercept, 'list_of_dfs': list_of_dfs, 'list_of_spline_input_features': list_of_spline_input_features}
        
        # compute the penalty matrix
        P = _get_P_from_design_matrix(structured_matrix, data)
        
        # add content to the dicts to be returned
        meta_datadict[param]['structured'] = structured_matrix.values
        parsed_formula_contents[param]['struct_shapes'] = structured_matrix.shape[1]
        parsed_formula_contents[param]['P'] = P
        parsed_formula_contents[param]['deep_models_dict'] = dict()
        parsed_formula_contents[param]['deep_shapes'] = dict()
        
        # if there are unstructured terms in the formula (returned from split_formula)
        if unstructured_terms:
            
            # for each unstructured term of the unstructured part of the formula
            for term in unstructured_terms:
                
                # get the feature name as input to each term
                term_split = term.split('(')
                net_name = term_split[0]
                feature_names = term_split[1].split(')')[0]
                
                # create a list of feature names if there are multiple inputs in term
                feature_names_list = feature_names.split(',')
                
                # and create the unstructured data
                unstructured_data = data[feature_names_list]
                unstructured_data = unstructured_data.to_numpy()
                meta_datadict[param][net_name] = unstructured_data
                
                # new addition - what is done here??
                if isinstance(deep_models_dict[net_name]['model'],str):
                    parsed_formula_contents[param]['deep_models_dict'][net_name]= eval(deep_models_dict[net_name]['model'])
                    
                else:
                    parsed_formula_contents[param]['deep_models_dict'][net_name]= deep_models_dict[net_name]['model']
                    
                parsed_formula_contents[param]['deep_shapes'][net_name] = deep_models_dict[net_name]['output_shape']
                
    return parsed_formula_contents, meta_datadict,  dm_info_dict

