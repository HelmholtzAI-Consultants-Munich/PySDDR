import numpy as np

'''
formulas = dict()

formulas['loc'] = '1 + s(x1) + s(x2) + dm1(images) + dm2(x2)'
'
formulas['scale'] = '1 + s(x)' 

meta_datadict = parse_formulas(formulas, data)

# datadict = {"structured": data, "dm1": data, "dm2": data}
        
'''

def parse_formulas(formulas, data):
    meta_datadict = dict()
    struct_list = []
    for param in formulas.keys():
        formula = formulas[param]
        # check formula formulation!
        if ' + ' not in formula:
            if len(formula) == 0
                print('Empty formula')
                # return intercept??
            else:
                # formula either only has one term OR formula wrongfully written --> HOW to check?
                if '+' in formula:
                    print('Rewrite your formula and make sure to leave a space before and after the plus sign')
                else:
                    # do the same as in for loop below - turn into function?
        else:
            formula_split = formula.split(' + ')
            for foruma_part in formula_split:
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
                    meta_datadict[param][model_name] = data[input_data_name]
                # linear part
                else:
                    l = ..
                    # first row of matrix should be the linear part
                    struct_list.append(l)
            # concatenate l,sx1,sx2 and put matrix into
            meta_datadict[param]['structured'] = np.concatenate(struct_list, dim=1)
            return meta_datadict


