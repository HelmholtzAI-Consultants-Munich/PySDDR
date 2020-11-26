from .utils import split_formula, get_info_from_design_matrix, get_P_from_design_matrix, orthogonalize_spline_wrt_non_splines, spline, compute_orthogonalization_pattern_deepnets
from patsy import dmatrix, build_design_matrices
import torch
import pandas as pd
import os

#import torch.nn as nn
#from model import TestNet
#import torchvision.models as models

class Prepare_Data(object):
    '''
    The Prepare_Data class parses the formulas defined by the user. This class includes fit and transform functions, which parses all information necessary to initialize the sddr network and
    also prepares the data by calculating penalty matrices (multiplied by the smoothing parameters lambda, which are computed from degrees of freedom) and orthogonalizing the non-linear (e.g. splines) wrt to the linear part of the formula.
    Parameters
    ----------
        formulas: dictionary
            A dictionary with keys corresponding to the parameters of the distribution defined by the user and values
            to strings defining the formula for each distribution, e.g. formulas['loc'] = '~ 1 + spline(x1, bs="bs", df=9) + dm1(x2)'.
        deep_models_dict: dictionary
            A dictionary where keys are model names and values are dicts with model architecture and output shapes.
        degrees_of_freedom: int or list of ints
            Degrees from freedom from which the smoothing parameter lambda is computed.
            Either a single value for all penalities of all splines, or a list of values, each for one of the splines that appear in the formula.

    Attributes
    -------
        formulas: dictionary
            A dictionary with keys corresponding to the parameters of the distribution defined by the user and values
            to strings defining the formula for each distribution, e.g. formulas['loc'] = '~ 1 + spline(x1, bs="bs", df=9) + dm1(x2)'.
        deep_models_dict: dictionary
            A dictionary where keys are model names and values are dicts with model architecture and output shapes.
        degrees_of_freedom: int or list of ints
            Degrees from freedom from which the smoothing parameter lambda is computed.
            Either a single value for all penalities of all splines, or a list of values, each for one of the splines that appear in the formula.
        network_info_dict: dictionary
            A dictionary where keys are the distribution's parameter names and values are dicts. The keys of these dicts
            will be: 'struct_shapes', 'P', 'deep_models_dict' and 'deep_shapes' with corresponding values for each distribution
            paramter, i.e. given formula (shapes of structured parts, penalty matrix, a dictionary of the deep models' arcitectures
            used in the current formula and the output shapes of these deep models)
        formula_terms_dict: dictionary
            A dictionary where keys are the distribution's parameter names and values are dicts containing the structured term of the formula (e.g. x1 + s(x2)), the list of unstructured terms (e.g. [d(x1),d(x2)])
            as well as a dictionary which maps the feature names to the unstructured terms (e.g. {"d(x1,x2):[x1,x2]"}).
        dm_info_dict: dictionary
            A dictionary where keys are the distribution's parameter names and values are dictionaries containing information of spline and non-spline terms (information: the corresponding slice in the formula, the term name and its input features).
        structured_matrix_design_info: dictionary
            A dictionary where keys are the distribution's parameter names and value is the design info of the patsy design matrix constructed for the structured part of the formula.
        data_range: list of integers
            Stored the maximum and minimum values of the input data set.
    '''
    def __init__(self, formulas, deep_models_dict, degrees_of_freedom, verbose=False):
        
        self.formulas = formulas
        self.deep_models_dict = deep_models_dict
        self.degrees_of_freedom = degrees_of_freedom

        self.network_info_dict = dict()
        self.formula_terms_dict = dict()

        #parse the content of the formulas for each parameter
        for param in formulas.keys():

            # split the formula into structured and unstructured parts
            structured_term, unstructured_terms = split_formula(self.formulas[param], list(self.deep_models_dict.keys()))

            # if there is not structured part create a null model
            if not structured_term:
                structured_part = '~0'

            # print the results of the splitting if verbose is set
            if verbose:
                print('results from split formula')
                print(structured_term)
                print(unstructured_terms)

            # initialize network_info_dict contains all information necessary to initialize the sddr_net
            self.network_info_dict[param] = dict()
            self.network_info_dict[param]['deep_models_dict'] = dict()
            self.network_info_dict[param]['deep_shapes'] = dict()
            self.network_info_dict[param]['orthogonalization_pattern'] = dict()
            
            
            # formula_terms_dict contains the splitted formula of structured and unstructured part as well as the names of the features are input to the different neural networks
            self.formula_terms_dict[param] = dict()
            self.formula_terms_dict[param]["structured_term"] = structured_term
            self.formula_terms_dict[param]["unstructured_terms"] = unstructured_terms
            self.formula_terms_dict[param]['net_feature_names'] = dict()
            
            # if there are unstructured terms in the formula (returned from split_formula)
            if unstructured_terms:

                # for each unstructured term of the unstructured part of the formula
                for term in unstructured_terms:
                    # get the feature name as input to each term
                    term_split = term.split('(')
                    net_name = term_split[0]
                    net_feature_names = term_split[1].split(')')[0].split(',')
                    # store deep models given by the user in a deep model dict that corresponds to the parameter in which this deep model is used
                    # if the deeps models are given as string, evaluate the expression first
                    if isinstance(self.deep_models_dict[net_name]['model'], str):
                        self.network_info_dict[param]['deep_models_dict'][net_name] = eval(self.deep_models_dict[net_name]['model'])
                    else:
                        self.network_info_dict[param]['deep_models_dict'][net_name] = self.deep_models_dict[net_name]['model']

                    self.network_info_dict[param]['deep_shapes'][net_name] = self.deep_models_dict[net_name]['output_shape']
                    
                    
                    self.formula_terms_dict[param]['net_feature_names'][net_name] = net_feature_names


    def fit(self,data):
        """
        Compute the penalty matrix, fits splines and stores information on the data, e.g. information on spline and non-spline terms.
        Computes orthogonalization pattern that defines which deep network features are orthogonalized w.r.t which structured terms.

        Parameters
        ----------
            data: Pandas.DataFrame
                input data (X)

        Returns
        -------

        """

        self.structured_matrix_design_info = dict()
        self.data_range = [data.min(),data.max()] # used in predict function
        self.P = dict()

        for param in self.formulas.keys():

            dfs = self.degrees_of_freedom[param]

            # create the structured matrix from the structured part of the formula - based on patsy
            structured_matrix = dmatrix(self.formula_terms_dict[param]["structured_term"], data, return_type='dataframe')
            self.structured_matrix_design_info[param] = structured_matrix.design_info

            # compute the penalty matrix and add content to the dicts to be returned
            self.P[param] = get_P_from_design_matrix(structured_matrix, dfs)  
            self.network_info_dict[param]['struct_shapes'] = structured_matrix.shape[1]
    '''
    def set_structured_matrix_design_info(self, structured_matrix_design_info):
        self.structured_matrix_design_info = structured_matrix_design_info
    
    def set_data_range(self, data_range):
        self.data_range = data_range
    '''
    def get_penalty_matrix(self, device):
        ''' Return penalty matrix as a torch and cast to device '''
        P = self.P
        # this only needs to be done for the first epoch of training
        for param in P.keys():
            P[param] = torch.from_numpy(P[param]).float() # should have shape struct_shapes x struct_shapes, numpy array
            P[param] = P[param].to(device)
        return P
    
    
    def transform(self,data,clipping=False):
        """
        Build patsy design matrix for input data and orthogonalize structured non-linear (e.g. splines) part wrt linear part.

        Parameters
        ----------
            data: Pandas.DataFrame
                input data (X)

        Returns
        -------
            prepared_data: dictionary
                A dictionary where keys are the distribution's parameter names and values are dicts. The keys of these dicts
                will be: 'structured' and neural network names if defined in the formula of the parameter (e.g. 'dm1'). Their values
                are the data for the structured part (after orthogonalization) and unstructured terms of the SDDR model.
        """
        prepared_data = dict()
        self.dm_info_dict = dict()
        
        for param in self.formulas.keys():
            prepared_data[param] = dict()

            # create the structured matrix using the same specification of the spline basis 
            try:
                structured_matrix = build_design_matrices([self.structured_matrix_design_info[param]], data, return_type='dataframe')[0]
            except Exception as e:
                if clipping == True:
                    train_data_min = {}
                    train_data_max = {}
                    for name in self.data_range[0].index:
                        train_data_min[name]=self.data_range[0][name]
                        train_data_max[name]=self.data_range[1][name]
                    clipped_data = data.clip(lower=pd.Series(train_data_min),upper=pd.Series(train_data_max),axis=1)
                    structured_matrix = build_design_matrices([self.structured_matrix_design_info[param]], clipped_data, return_type='dataframe')[0]
                else:
                    raise Exception("Data should stay within the range of the training data. Please try clipping or manually set knots.")

            # get bool depending on if formula has intercept or not and degrees of freedom and input feature names for each spline
            spline_info, non_spline_info = get_info_from_design_matrix(structured_matrix, feature_names=data.columns)
            self.dm_info_dict[param] = {'spline_info': spline_info, 'non_spline_info': non_spline_info }
            
            #compute the orthogonalization patterns for the deep neural networks
            for net_name in self.network_info_dict[param]['deep_models_dict'].keys():
                net_feature_names = self.formula_terms_dict[param]['net_feature_names'][net_name]
                orthogonalization_pattern = compute_orthogonalization_pattern_deepnets(net_feature_names, 
                                                                                       spline_info, 
                                                                                       non_spline_info) 
                
                self.network_info_dict[param]['orthogonalization_pattern'][net_name] = orthogonalization_pattern
            
            # orthogonalize splines with respect to non-splines (including an intercept if it is there)
            orthogonalize_spline_wrt_non_splines(structured_matrix, spline_info, non_spline_info)

            # add content to the dicts to be returned
            prepared_data[param]["structured"] = torch.from_numpy(structured_matrix.values).float()

            for net_name in self.formula_terms_dict[param]['net_feature_names'].keys():
                net_feature_names = self.formula_terms_dict[param]['net_feature_names'][net_name]
                try:
                    prepared_data[param][net_name] = torch.from_numpy(data[net_feature_names].to_numpy()).float()
                except: #if it fails to convert to numpy as the data type is an object: store the data frame itself
                    prepared_data[param][net_name] = data[net_feature_names]

        return prepared_data