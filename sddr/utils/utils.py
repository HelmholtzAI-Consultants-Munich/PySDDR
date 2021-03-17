import numpy as np
import pandas as pd
import scipy as sp
import warnings
import os
from torch import nn
import torch
import parser
from .splines import spline, Spline


def checkups(params, formulas):
    """
    Checks if the user has given an available distribution, too many formulas or wrong parameters for the given distribution.

    Parameters
    ----------
        params : list of strings
            A list of strings of the parameters of the current distribution.
        formulas : dictionary
            A dictionary with keys corresponding to the parameters of the distribution defined by the user and values to strings 
            defining the formula for each distribution, e.g. formulas['loc'] = '~ 1 + spline(x1, bs="bs", df=9) + dm1(x2)'.
            
    Returns
    -------
        new_formulas : dictionary
            If the current distribution is available in the list of families, new_formulas holds a formula for each parameter of 
            the distribution.
            If a formula hasn't been given for a parameter, ~0 formula is set. If the current distribution is not available, 
            an empty dictionary is returned.  
    """
    new_formulas=dict()
    for param in params:
        if param in formulas:
            new_formulas[param] = formulas[param]
        # define an empty formula for parameters for which the user has not specified a formula
        else:
            warnings.simplefilter('always')
            warnings.warn(f'Parameter formula f{param} for distribution not defined. Creating a zero formula for it.', stacklevel=2)
            new_formulas[param] = '~0'
    return new_formulas



def split_formula(formula, net_names_list):
    """
    Splits the formula into two parts - the structured and unstructured part.

    Parameters
    ----------
        formula : string
            The formula to be split, e.g. '~ 1 + bs(x1, df=9) + dm1(x2, df=9)'.
        net_names_list : list of strings
            A list of all network names defined by the user.
            
    Returns
    -------
        structured_part : string
            A string holding only the structured part of the original formula.
        unstructured_terms: list of strings
            A list holding all the unstructured parts of the original formula.
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
    structured_term = '+'.join(structured_terms)
    return structured_term, unstructured_terms



def make_matrix_positive_semi_definite(A,machine_epsilon):
    """
    Transforms an input matrix such that it is semipositive definite.

    Parameters
    ----------
        A : numpy array
            Any input matrix.
        machine_epsilon : int
            Tolerance value (machine epsilon value).

    Returns
    -------
        A : numpy array
            A matrix that is semipositive definite.
    """
    #get smallest eigenvalue for a symmetric matrix and use some additional tolerance to ensure semipositive definite matrices
    min_eigen = min(np.linalg.eigh(A)[0]) - np.sqrt(machine_epsilon)

    # smallest eigenvalue negative = not semipositive definit
    if min_eigen < -1e-10:
        rho = 1 / (1 - min_eigen)
        A = rho * A + (1 - rho) * np.identity(A.shape[0])

        ## now check if it is really positive definite by recursively calling
        A = make_matrix_positive_semi_definite(A)
    return (A)



def df_fun(lam, d, hat1):
    """
    Calculates degrees of freedom from lambda.

    Parameters
    ----------
        lam : int
            Lambda value.
        d : int
            Vector of singular values from SVD.

    Returns
    -------
        df : int
            Degrees of Freedom.
    """
    if hat1:
        df = sum(1 / (1 + lam * d))
    else:
        df = 2 * sum(1 / (1 + lam * d)) - sum(1 / (1 + lam * d) ^ 2)
    return df




def df2lambda(dm, P, df, lam = None, hat1 = True, lam_max = 1e+15):
    """
    Calculates lambda from degrees of freedom (default) or degrees of freedom from lambda.

    Parameters
    ----------
        dm : patsy.dmatrix
            The design matrix for the structured part of the formula - computed by patsy.
        P: numpy-array
            The penalty matrix of the design matrix.
        df: int
            Degrees of Freedom. Can be "None" if lambda value is provided.
        lam: int, default None
            Lambda value.
        lam_max: int, default 1e+15
            Maximum value for lambda. Can be adjusted depending on needs.

    Returns
    -------
        df: int
            Degrees of Freedom.
        lam: int
            Lambda.
    """
    ## define tolerance value, here we use machine epsilon
    machine_epsilon = np.finfo(float).eps * 2

    ## throw exception if neither df nor lambda is given
    if df == None and lam == None:
        raise Exception('Either degrees of freedom or lambda has to be provided.')

    ## check if rank of design matrix is large enough for given df
    if df != None:
        rank_dm = np.linalg.matrix_rank(dm)
        if df >= rank_dm:
            warnings.simplefilter('always')
            warnings.warn("""df too large: Degrees of freedom (df = {0}) cannot be larger than the rank of the design matrix (rank = {1}). 
            Unpenalized base-learner with df = {1} will be used. Re-consider model specification.""".format(df,rank_dm), stacklevel=2)
            lam = 0
            return df, lam

    ## if lambda is given, but equal 0, return rank of design matrix as df
    if lam != None:
        if lam == 0:
            df = np.linalg.matrix_rank(dm)
            return df, lam

    ## otherwise compute df or lambda

    # avoid that XtX matrix is not (numerically) singular
    XtX = dm.T @ dm

    # make sure that A is numerically positiv semi-definit and also numerically symmetric
    A = XtX + P * 1e-15
    A = make_matrix_positive_semi_definite(A,machine_epsilon)
    Rm = sp.linalg.solve_triangular(sp.linalg.cholesky(A, lower=False), np.identity(XtX.shape[1]))

    # singular value decomposition and return vector (d) with singular values (might be possible to speed up if set 'hermitian = True')
    try:
        # try computation without computing u and vh
        d = np.linalg.svd((Rm.T @ P) @ Rm, compute_uv = False)
    except:
        # if unsuccessful try the same computation but compute u and vh as well
        d = np.linalg.svd((Rm.T @ P) @ Rm, compute_uv=True)[1]

    # if lambda given compute degrees of freedom
    if lam != None:
        df = df_fun(lam, d, hat1)
        return df, lam

    # else compute lambda from degrees of freedom through optimization
    if df >= len(d):
        lam = 0
        return df, lam

    df_for_lam_max = df_fun(lam_max, d, hat1)
    if (df_for_lam_max - df) > 0 and (df_for_lam_max - df) > np.sqrt(machine_epsilon):
        warnings.simplefilter('always')
        warnings.warn("""lambda needs to be larger than lambda_max = {0} for given df. Settign lambda to {0} leeds to an deviation from your df of {1}. 
        You can increase lambda_max in parameters. """.format(lam_max,df_for_lam_max - df), stacklevel=2)
        lam = lam_max
        return df, lam

    lam = sp.optimize.brentq(lambda l: df_fun(l, d, hat1) - df, 0, lam_max)
    if abs(df_fun(lam, d, hat1) - df) > np.sqrt(machine_epsilon):
        warnings.simplefilter('always')
        warnings.warn("""estimated df differ from given df by {0} """.format(df_fun(lam, d, hat1) - df), stacklevel=2)

    return df, lam




def _get_penalty_matrix_from_factor_info(factor_info):
    '''
    Extracts the penalty matrix from a factor_info object if the factor info object is a spline.
    Explanation: "spline" is a stateful pasty transform. After computation of the design matrix these stateful transforms are stored in the factor infos of the design matrix. In this function we extract this object and obtain the penalty matrix that corresponds to this spline.

    Parameters
    ----------
        factor_info: patsy factor info object.
             
    Returns
    -------
        P: numpy-array or False
            Penalty matrix of the spline term or False.
    '''
    factor = factor_info.factor

    # if factor is not a spline, there is not penalty matrix
    if 'spline' not in factor.name():
        P = False
        return P

    # a factor can be nested e.g. Spline(center(x)). The outer objects (e.g. Spline) are the last in the list 'pass_bins'
    outer_objects_in_factor = factor_info.state['pass_bins'][-1]

    # use last=outermost object in factor. should only have a single element in the set. Explanation: the 'pass_bins' list contains tuples. e.g. if an object is Spline(center(x1)+center(x2)) then 'pass_bins' is a list with two tuples,
    # where the first tuple containts the name of the two center objects. There should only be one element in the outer tuple and this is extracted here(we will check later, that this tuple contains indeed only one element)
    obj_name = next(iter(outer_objects_in_factor))

    # obtain the actal statefull transform object that corresponds to the extracted object name
    obj = factor_info.state['transforms'][obj_name]

    # check if the tuple indeed contained only one element and that the obtained object is a spline. If both is true obatain and return the penalty matrix of this spline.
    if (len(outer_objects_in_factor)==1) and isinstance(obj, Spline):
        P = obj.penalty_matrices
        return P
    # factor is not a spline, so there is not penalty matrix
    else:
        P = False
        return P 




def get_P_from_design_matrix(dm, dfs):
    """
    Computes and returns the penalty matrix that corresponds to a patsy design matrix. The penalties are multiplied by the regularization parameters lambda computed from given degrees of freedom.
    The result is a single block diagonal penalty matrix that combines the penalty matrices of each term in the formula that was used to create the design matrix. Only smooting splines terms have a non-zero penalty matrix.
    The degrees of freedom can either be given as a single value, then all individual penalty matrices are multiplied with a single lambda. Or they can be given as a list, then all (non-zero) penalty matrices are multiplied by different lambdas. The multiplication is in the order of the terms in the formula.
    
    Parameters
    ----------
        dm: patsy.dmatrix
            The design matrix for the structured part of the formula - computed by patsy.
        dfs: int or list of ints
            Degrees from freedom from which the smoothing parameter lambda is computed.
            Either a single value for all penalities of all splines, or a list of values, each for one of the splines that appear 
            in the formula.

    Returns
    -------
        big_P: numpy array
            The penalty matrix of the design matrix.
    """
    factor_infos = dm.design_info.factor_infos
    terms = dm.design_info.terms
    
    big_P = np.zeros((dm.shape[1],dm.shape[1]))
    
    spline_counter = 0
    
    for term in terms:
        dm_term_name = term.name()

        # get the slice object for this term (corresponding to start and end index in the design matrix)
        slice_of_term = dm.design_info.term_name_slices[dm_term_name]

        # currently we only use smoothing for 1D, in the future we also want to add smoothing for tensorproducts
        if len(term.factors) == 1:
            factor_info = factor_infos[term.factors[0]]
            num_columns = factor_info.num_columns
            
            P = _get_penalty_matrix_from_factor_info(factor_info)
                
            if P is not False:
                df = dfs[spline_counter] if type(dfs) == list else dfs
                dm_spline = dm.iloc[:,slice_of_term]

                # Regularization parameters are given in degrees of freedom. Here they are converted to lambda.
                df_lam = df2lambda(dm_spline, P[0], df)
                big_P[slice_of_term,slice_of_term] = P[0]*df_lam[1]
                spline_counter += 1
    return big_P




def _get_input_features_for_functional_expression(functional_expression : str, feature_names : list):
    '''
    Parses variables from a functional expression using the python parser

    Parameters
    ----------
        functional_expression: string
            Functional expression from which to extract the input features like "spline(x1,x2, bs="bs", df=4, degree=3)".
            
        feature_names: set
            Set of all possible feature names in the data set like [x1,x2,x3,x4,x5].
            
    Returns
    -------
        input_features: list
            List of feature names that appear as input in functional_expression. here in the example ["x1","x2"].
    '''
    # co names are local variables of functions in a python expression
    co_names = parser.expr(functional_expression).compile().co_names
    co_names_set = set(co_names)
    input_features = list(co_names_set.intersection(set(feature_names)))
    return input_features


def _get_all_input_features_for_term(term, feature_names):
    '''
    Extracts all feature names that appear in a patsy term. For this it loops through all factors and uses then a python code paser to extract input variables.

    Parameters
    ----------
        term: patsy term object
            Patsy term object for which the feature names should be extracted.
            
        feature_names: list
            List of all possible feature names in the data set like [x1,x2,x3,x4,x5].
            
    Returns
    -------
        input_features_term: list
            List of feature names that appear in the patsy term. e.g. for a term x1:spline(x2, bs="bs", df=4, degree=3) -> 
            ["x1","x2"].
    '''
    factors = term.factors
    input_features_term = set()
    for factor in factors:
        factor_name = factor.name()
        input_features_factor = _get_input_features_for_functional_expression(factor_name, list(feature_names))
        input_features_term = input_features_term.union(set(input_features_factor))
        
    input_features_term = list(input_features_term)
    return input_features_term




def get_info_from_design_matrix(structured_matrix, feature_names):
    """
    Parses the formulas defined by the user and returns a dict of dicts which can be fed into SDDR network

    Parameters
    ----------
        structured_matrix: patsy.dmatrix
            The design matrix for the structured part of the formula - computed by patsy.
        feature_names: list of strings
            A list of feature names (column names of the input data set).

    Returns
    -------
        spline_info: dictionary
            A dictionary containing information of the spline terms: the respective slice in the formula, the name of the spline 
            and the feature names of the spline.
        non_spline_info: dictionary
            A dictionary containing information of the terms that are not splines: the respective slice in the formula, the name 
            of the term and the feature name of the term.
    """
    spline_info = {'list_of_spline_slices': [],
                   'list_of_spline_input_features': [],
                   'list_of_term_names' : []}
    
    non_spline_info = {'list_of_non_spline_slices': [],
                       'list_of_non_spline_input_features': [],
                       'list_of_term_names' : []}
    
    for term in structured_matrix.design_info.terms:
        dm_term_name = term.name()

        # get the feature names sent as input to each spline
        feature_names_spline = _get_all_input_features_for_term(term, feature_names)

        # get the slice object for this term (corresponding to start and end index in the designmatrix)
        slice_of_term = structured_matrix.design_info.term_name_slices[dm_term_name] 

        # append to lists
        if 'spline' in dm_term_name:
            spline_info['list_of_spline_input_features'].append(feature_names_spline)
            spline_info['list_of_spline_slices'].append(slice_of_term)
            spline_info['list_of_term_names'].append(dm_term_name)
        else:
            non_spline_info['list_of_non_spline_input_features'].append(feature_names_spline)
            non_spline_info['list_of_non_spline_slices'].append(slice_of_term)
            non_spline_info['list_of_term_names'].append(dm_term_name)
            
    return spline_info, non_spline_info




def _orthogonalize(constraints, X):
    """
    Orthogonalize spline terms with respect to non spline terms.

    Parameters
    ----------
        constraints: numpy array
            constraint matrix, non spline terms
        X: numpy array
            spline terms

    Returns
    -------
        constrained_X: numpy array
            orthogonalized spline terms
    """
    Q, _ = np.linalg.qr(constraints) # compute Q
    Projection_Matrix = np.matmul(Q,Q.T)
    temp=np.matmul(Projection_Matrix,X)
    constrained_X = X - np.matmul(Projection_Matrix,X)
    return constrained_X


def orthogonalize_spline_wrt_non_splines(structured_matrix, 
                                         spline_info, 
                                         non_spline_info):
    '''
    Changes the structured matrix by orthogonalizing all spline terms with respect to all non spline terms.
    Orthogonalization of spline term is only with respect to the non-spline terms that contain a subset of the features that are input to the spline (inlcuding the intercept). E.g. spline(x3, bs='bs', df=9, degree=3) is orthogonalized with respect to the intercept and x3. If any terms x2, x4 ... appear they are ignored in this orthogonalization.
    
    The change on the structured matrix is done inplace!
    
    Parameters
    ----------
        structured_matrix: patsy.dmatrix
            The design matrix for the structured part of the formula - computed by patsy
        spline_info: dict
            dictionary with keys list_of_spline_slices and list_of_spline_input_features. As produced by
            get_info_from_design_matrix
        non_spline_info: dict
            dictionary with keys list_of_non_spline_slices and list_of_non_spline_input_features. 
    '''
    
    for spline_slice, spline_input_features in zip(spline_info['list_of_spline_slices'], 
                                                   spline_info['list_of_spline_input_features']):
        
        X = structured_matrix.iloc[:,spline_slice]
        # construct constraint matrix
        constraints = []
        for non_spline_slice, non_spline_input_features in zip(non_spline_info['list_of_non_spline_slices'], non_spline_info['list_of_non_spline_input_features']):
            if set(non_spline_input_features).issubset(set(spline_input_features)):
                constraints.append(structured_matrix.iloc[:,non_spline_slice].values)

        if len(constraints)>0:
            constraints = np.concatenate(constraints,axis=1)
            constrained_X = _orthogonalize(constraints, np.array(X))
            structured_matrix.iloc[:,spline_slice] = constrained_X
        
            
def compute_orthogonalization_pattern_deepnets(net_feature_names, 
                                               spline_info, 
                                               non_spline_info):
    '''
    Computes the orthogonalization pattern that tells with respect to which structured terms the features of a deep neural network should be orthogonalized. Returned is a list of slices which is then used in the orthogonalization to slice the design matrix for the strucutred part of the formula.
    Orthogonalization of deep net term is only with respect to the structured terms that contain a subset of the features that are input to the deep neural network (inlcuding the intercept). E.g. d1(x3) is orthogonalized with respect to the intercept,x3 and a spline that has as only input x3. If any terms x2, x4 or a spline with another input than x2 e.g. spline(x1,x3) or spline(x1) appear they are ignored in this orthogonalization.
    
    Parameters
    ----------
        net_feature_names: list of strings
            list of names of input features to the deep neural network
        spline_info: dict
            dictionary with keys list_of_spline_slices and list_of_spline_input_features. As produced by 
            get_info_from_design_matrix
        non_spline_info: dict
            dictionary with keys list_of_non_spline_slices and list_of_non_spline_input_features. 
            
    Returns
    -------
        orthogonalization_pattern: list of slice objects
            For each term in the design matrix wrt that the deep neural network should be orthogonalized there is 
            a slice in the list.
    '''
    
    
    orthogonalization_pattern = []
    for non_spline_slice, non_spline_input_features in zip(non_spline_info['list_of_non_spline_slices'],
                                                           non_spline_info['list_of_non_spline_input_features']):
        
        if set(non_spline_input_features).issubset(set(net_feature_names)):
            orthogonalization_pattern.append(non_spline_slice)
            
    for spline_slice, spline_input_features in zip(spline_info['list_of_spline_slices'],
                                                   spline_info['list_of_spline_input_features']):
        
        if set(spline_input_features).issubset(set(net_feature_names)):
            orthogonalization_pattern.append(spline_slice)
    return orthogonalization_pattern