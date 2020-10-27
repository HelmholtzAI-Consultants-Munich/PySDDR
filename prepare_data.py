from utils import _split_formula, _get_info_from_design_matrix, _get_P_from_design_matrix, _orthogonalize_spline_wrt_non_splines, spline
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

        for param in formulas.keys():

            # split the formula into structured and unstructured parts
            structured_part, unstructured_terms = _split_formula(self.formulas[param], list(self.deep_models_dict.keys()))

            # if there is not structured part create a null model
            if not structured_part:
                structured_part = '~0'

            # print the results of the splitting if verbose is set
            if verbose:
                print('results from split formula')
                print(structured_part)
                print(unstructured_terms)

            # save network information
            self.network_info_dict[param] = dict()
            self.network_info_dict[param]['deep_models_dict'] = dict()
            self.network_info_dict[param]['deep_shapes'] = dict()

            self.formula_terms_dict[param] = dict()
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

            
            self.formula_terms_dict[param]["structured_part"] = structured_part
            self.formula_terms_dict[param]["unstructured_terms"] = unstructured_terms


    def fit(self,data):
        
        self.structured_part_data = dict()
        
        for param in self.formulas.keys():
            self.structured_part_data[param] = dict()

            dfs = self.degrees_of_freedom[param]

            # create the structured matrix from the structured part of the formula - based on patsy
            self.structured_matrix = dmatrix(self.formula_terms_dict[param]["structured_part"], data, return_type='dataframe')

            # get bool depending on if formula has intercept or not and degrees of freedom and input feature names for each spline
            spline_info, non_spline_info = _get_info_from_design_matrix(self.structured_matrix, feature_names=data.columns)
            self.dm_info_dict[param] = spline_info

            # compute the penalty matrix
            P = _get_P_from_design_matrix(self.structured_matrix, data, dfs)

            # orthogonalize splines with respect to non-splines (including an intercept if it is there)
            _orthogonalize_spline_wrt_non_splines(self.structured_matrix, spline_info, non_spline_info)

            # add content to the dicts to be returned
            self.structured_part_data[param] = self.structured_matrix.values
            self.network_info_dict[param]['struct_shapes'] = self.structured_matrix.shape[1]
            self.network_info_dict[param]['P'] = P
                               
        self.unstructured_part_data = dict()       
        for param in self.formulas.keys():
            self.unstructured_part_data[param] = dict()
            
            for net_name in self.formula_terms_dict[param]['net_feature_names'].keys():
                net_feature_names = self.formula_terms_dict[param]['net_feature_names'][net_name]
                self.unstructured_part_data[param][net_name] = data[net_feature_names].to_numpy()

    def get_batch_train(self,structured_index, unstructured_data):

        batch_data = dict()
        for param in self.formulas.keys():
            batch_data[param] = dict()
            batch_data[param]["structured"] = self.structured_part_data[param][structured_index]

            for net_name in self.formula_terms_dict[param]['net_feature_names'].keys():
                net_feature_names = self.formula_terms_dict[param]['net_feature_names'][net_name]
                batch_data[param][net_name] = self.unstructured_part_data[param][net_name][structured_index] 

        return batch_data
    

    def get_batch_predict(self,data):
        #use:
        #from patsy import dmatrix, build_design_matrices
        # build_design_matrices([mat.design_info], new_data)[0]
        
        structured_part_data_predict = dict()
        
        for param in self.formulas.keys():
            structured_part_data_predict[param] = dict()
            
            # create the structured matrix using the same specification of the spline basis
            structured_matrix_predict = build_design_matrices([self.structured_matrix.design_info],data,return_type='dataframe')[0]
            
            # get bool depending on if formula has intercept or not and degrees of freedom and input feature names for each spline
            spline_info, non_spline_info = _get_info_from_design_matrix(structured_matrix_predict, feature_names=data.columns)
            
            # orthogonalize splines with respect to non-splines (including an intercept if it is there)
            _orthogonalize_spline_wrt_non_splines(structured_matrix_predict, spline_info, non_spline_info)

            # add content to the dicts to be returned
            structured_part_data_predict[param] = structured_matrix_predict.values
                
                
        pred_data = dict()
        for param in self.formulas.keys():
            pred_data[param] = dict()
            pred_data[param]["structured"] = torch.from_numpy(structured_part_data_predict[param]).float()

            for net_name in self.formula_terms_dict[param]['net_feature_names'].keys():
                net_feature_names = self.formula_terms_dict[param]['net_feature_names'][net_name]
                pred_data[param][net_name] = torch.from_numpy(data[net_feature_names].to_numpy()).float()

        return pred_data
        