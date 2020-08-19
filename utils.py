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
    for param in formulas.keys():
        formula = formulas[param]
        # check formula formulation!
        if ' + ' not in formula:
            if len(formula) == 0
                print('Empty formula')
                # return intercept??
            else:
                # formula either only has one term OR formula wrongfully written --> HOW to check?
        else:
            formula_split = formula.split(' + ')
            for foruma_part in formula_split:
                if formula_part[0] == 's':
                    # perform smoothing
                    sx1 = s[data['x1']]
                    sx2 = s[data['x2']]
                # unstructured part    
                elif formula_part[0] == 'd':
                    model_name = formula_part[0].split('(')[0]
                    input_data = formula_part[0].split('(')[-1].split(')')[0]
                    meta_datadict[param][model_name] = data[input_data]
                # linear part
                else:
                    l = ..
            # concatenate l,sx1,sx2 and put matrix into
            meta_datadict[param]['structured'] = struct_matrix
            return meta_datadict