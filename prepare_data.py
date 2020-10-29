from utils import split_formula, get_info_from_design_matrix, get_P_from_design_matrix, orthogonalize_spline_wrt_non_splines, spline
from patsy import dmatrix, build_design_matrices
import torch

class Prepare_Data(object):

    def __init__(self, formulas, deep_models_dict, degrees_of_freedom, verbose=False):
        
        self.formulas = formulas
        self.deep_models_dict = deep_models_dict
        self.degrees_of_freedom = degrees_of_freedom

        self.network_info_dict = dict()
        self.dm_info_dict = dict()
        self.formula_terms_dict = dict()

        #parse the content of the formulas for each parameter
        for param in formulas.keys():

            # split the formula into structured and unstructured parts
            structured_part, unstructured_terms = split_formula(self.formulas[param], list(self.deep_models_dict.keys()))

            # if there is not structured part create a null model
            if not structured_part:
                structured_part = '~0'

            # print the results of the splitting if verbose is set
            if verbose:
                print('results from split formula')
                print(structured_part)
                print(unstructured_terms)

            # initialize dictionaries
            # initialize network_info_dict contains all information necessary to initialize the sddr_net
            self.network_info_dict[param] = dict()
            self.network_info_dict[param]['deep_models_dict'] = dict()
            self.network_info_dict[param]['deep_shapes'] = dict()
            
            # formula_terms_dict contains the splitted formula of structured and unstructured part as well as the nemaes of the features are input to the different neural networks
            self.formula_terms_dict[param] = dict()
            self.formula_terms_dict[param]["structured_part"] = structured_part
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
        self.structured_matrix_design_info = dict()
        
        self.data_info = [data.min(level=0),data.max(level=0)] # used in predict function
        
        for param in self.formulas.keys():

            dfs = self.degrees_of_freedom[param]

            # create the structured matrix from the structured part of the formula - based on patsy
            structured_matrix = dmatrix(self.formula_terms_dict[param]["structured_part"], data, return_type='dataframe')
            self.structured_matrix_design_info[param] = structured_matrix.design_info

            # get bool depending on if formula has intercept or not and degrees of freedom and input feature names for each spline
            spline_info, non_spline_info = get_info_from_design_matrix(structured_matrix, feature_names=data.columns)
            self.dm_info_dict[param] = {'spline_info': spline_info, 'non_spline_info': non_spline_info }

            # compute the penalty matrix
            P = get_P_from_design_matrix(structured_matrix, data, dfs)
            
            # add content to the dicts to be returned
            self.network_info_dict[param]['struct_shapes'] = structured_matrix.shape[1]
            self.network_info_dict[param]['P'] = P    
            
    def transform(self,data):
        
        prepared_data = dict()
        train_data_min,train_data_max = self.data_info
        
        for param in self.formulas.keys():
            prepared_data[param] = dict()
            
            # create the structured matrix using the same specification of the spline basis 
            try:
                structured_matrix = build_design_matrices([self.structured_matrix_design_info[param]], data, return_type='dataframe')[0]
            except Exception as e:
                structured_matrix = build_design_matrices([self.structured_matrix_design_info[param]], data.clip(train_data_min,train_data_max), return_type='dataframe')[0]
                print('Data should stay within the range of the training data, they are clipped here.')
            
            spline_info = self.dm_info_dict[param]['spline_info']
            non_spline_info = self.dm_info_dict[param]['non_spline_info']
            
            # orthogonalize splines with respect to non-splines (including an intercept if it is there)
            orthogonalize_spline_wrt_non_splines(structured_matrix, spline_info, non_spline_info)

            # add content to the dicts to be returned
            prepared_data[param]["structured"] = torch.from_numpy(structured_matrix.values).float()

            for net_name in self.formula_terms_dict[param]['net_feature_names'].keys():
                net_feature_names = self.formula_terms_dict[param]['net_feature_names'][net_name]
                prepared_data[param][net_name] = torch.from_numpy(data[net_feature_names].to_numpy()).float()

        return prepared_data
        
    def get_item_from_data(self,index, data):

        item = dict()
        for param in self.formulas.keys():
            item[param] = dict()
            item[param]["structured"] = data[param]['structured'][index]

            for net_name in self.formula_terms_dict[param]['net_feature_names'].keys():
                net_feature_names = self.formula_terms_dict[param]['net_feature_names'][net_name]
                item[param][net_name] = data[param][net_name][index] 

        return item
    
    
"""
    Documentation of old parse_formulas function:
    =============================================
    
    
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
        degrees_of_freedom: dict
            A dictionary where keys are the name of the distribution parameter (e.g. eta,scale) and values 
            are either a single smooting parameter for all penalities of all splines for this parameter, or a list of smooting parameters, each for one of the splines that appear in the formula for this parameter
            
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