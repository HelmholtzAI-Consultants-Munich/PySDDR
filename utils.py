import numpy as np
import pygam

'''
formulas = dict()

formulas['loc'] = '1 + s(x1) + s(x2) + dm1(images) + dm2(x2)'
'
formulas['scale'] = '1 + s(x)' 

meta_datadict = parse_formulas(formulas, data)
# datadict = {"structured": data, "dm1": data, "dm2": data}
  
'''

def checkups(family, formulas, cur_distribution):
    if cur_distribution not in family.keys():
        print('Distribution not in family of distributions!')
        exit    
    else:
        #if len(formulas) > len(family[cur_distribution]):
        # check either if too many formulas have been given and drop them or if wrong parameter names have been given
        new_formulas=dict()
        for param in family[cur_distribution].keys():
            if param in formulas:
                new_formulas[param] = formulas[param]
            else:
                print('parameter', param,'for distribution not defined')
                new_formulas[param] = '~0'
        return new_formulas
            
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
    
def preprocess(formulas, family, data, cur_distribution, list_of_nn_names):
    # perform checks on given distribution name, parameter names and number of formulas given
    formulas = checkups(family, formulas, cur_distribution)
    meta_datadict = dict()
    struct_list = []
    for param in formulas.keys():
        formula_dict = parse_formula(formulas[param], data, param, list_of_nn_names)
        meta_datadict[param] = formula_dict 
    return meta_datadict


