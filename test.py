import unittest

from patsy import dmatrix
import statsmodels.api as sm
from utils import parse_formulas, spline
import numpy as np
from torch import nn

import unittest

from patsy import dmatrix
import statsmodels.api as sm
from utils import parse_formulas, spline, Family
import numpy as np
from torch import nn

class Testparse_formulas(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super(Testparse_formulas, self).__init__(*args,**kwargs)
        # load data
        iris = sm.datasets.get_rdataset('iris').data
        self.x = iris.rename(columns={'Sepal.Length':'x1','Sepal.Width':'x2','Petal.Length':'x3','Petal.Width':'x4','Species':'y'})
        
        
    def test_patsyfreedummytest_parse_formulas(self):
        """
        Test if parse_formulas without assuming pasty is correct
        """
        # define formulas

        formulas = dict()
        formulas['loc'] = '~1'
        formulas['scale'] = '~1'

        # define distributions and network names
        cur_distribution = 'Normal'
        family = Family(cur_distribution)

        deep_models_dict = dict()

        #call parse_formulas
        parsed_formula_content, meta_datadict = parse_formulas(family, formulas, self.x, deep_models_dict)       
        
        ground_truth = np.ones([len(self.x),1])
        #test if shapes of design matrices and P are as correct
        self.assertTrue((meta_datadict['loc']['structured'] == ground_truth).all())
        self.assertTrue((meta_datadict['scale']['structured'] == ground_truth).all())
        self.assertTrue((meta_datadict['loc']['structured'].shape == ground_truth.shape),'shape missmatch')
        self.assertTrue((meta_datadict['scale']['structured'].shape == ground_truth.shape),'shape missmatch')
        self.assertEqual(parsed_formula_content["loc"]['struct_shapes'], 1)
        self.assertEqual(parsed_formula_content["loc"]['P'].shape, (1, 1))
        self.assertEqual(parsed_formula_content["loc"]['P'], 0)
        
        self.assertEqual(parsed_formula_content["scale"]['struct_shapes'], 1)
        self.assertEqual(parsed_formula_content["scale"]['P'].shape, (1, 1))
        self.assertEqual(parsed_formula_content["scale"]['P'], 0)
        
    def test_structured_parse_formulas(self):
        """
        Test if linear model is correctly processed in parse_formulas
        """
        # define formulas

        formulas = dict()
        formulas['loc'] = '~1'
        formulas['scale'] = '~1 + x1'

        # define distributions and network names
        cur_distribution = 'Normal'
        family = Family(cur_distribution)

        deep_models_dict = dict()

        #call parse_formulas
        parsed_formula_content, meta_datadict = parse_formulas(family, formulas, self.x, deep_models_dict)       
        
        ground_truth_loc = dmatrix(formulas['loc'], self.x, return_type='dataframe').to_numpy()
        ground_truth_scale = dmatrix(formulas['scale'], self.x, return_type='dataframe').to_numpy()
        #test if shapes of design matrices and P are as correct
        self.assertTrue((meta_datadict['loc']['structured'] == ground_truth_loc).all())
        self.assertTrue((meta_datadict['scale']['structured'] == ground_truth_scale).all())
        self.assertTrue((meta_datadict['loc']['structured'].shape == ground_truth_loc.shape),'shape missmatch')
        self.assertTrue((meta_datadict['scale']['structured'].shape == ground_truth_scale.shape),'shape missmatch')
        self.assertEqual(parsed_formula_content["loc"]['struct_shapes'], 1)
        self.assertEqual(parsed_formula_content["loc"]['P'].shape, (1, 1))
        self.assertTrue((parsed_formula_content["loc"]['P']==0).all())
        self.assertEqual(parsed_formula_content["scale"]['struct_shapes'], 2)
        self.assertEqual(parsed_formula_content["scale"]['P'].shape, (2, 2))
        self.assertTrue((parsed_formula_content["scale"]['P']==0).all())
        
    def test_unstructured_parse_formulas(self):
        """
        Test if parse_formulas is correctly dealing with NNs
        """
        # define formulas

        formulas = dict()
        formulas['loc'] = '~1 + d1(x2,x1,x3)'
        formulas['scale'] = '~1 + x1 + d2(x1)'

        # define distributions and network names
        cur_distribution = 'Normal'
        family = Family(cur_distribution)
        
        deep_models_dict = dict()
        deep_models_dict['d1'] = {'model': nn.Sequential(nn.Linear(1,15)), 'output_shape': 42}
        deep_models_dict['d2'] = {'model': nn.Sequential(nn.Linear(1,15)), 'output_shape': 42}

        #call parse_formulas
        parsed_formula_content, meta_datadict = parse_formulas(family, formulas, self.x, deep_models_dict)       

        ground_truth_loc = dmatrix('~1', self.x, return_type='dataframe').to_numpy()
        ground_truth_scale = dmatrix('~1 + x1', self.x, return_type='dataframe').to_numpy()
        #test if shapes of design matrices and P are as correct
        self.assertTrue((meta_datadict['loc']['structured'] == ground_truth_loc).all())
        self.assertTrue((meta_datadict['scale']['structured'] == ground_truth_scale).all())
        self.assertTrue((meta_datadict['loc']['structured'].shape == ground_truth_loc.shape),'shape missmatch')
        self.assertTrue((meta_datadict['scale']['structured'].shape == ground_truth_scale.shape),'shape missmatch')
        
        self.assertTrue((meta_datadict['loc']['d1'] == self.x[['x2','x1','x3']].to_numpy()).all())
        self.assertTrue((meta_datadict['loc']['d1'].shape == self.x[['x2','x1','x3']].shape),'shape missmatch for neural network input')
        self.assertTrue((meta_datadict['scale']['d2'] == self.x[['x1']].to_numpy()).all())
        self.assertTrue((meta_datadict['scale']['d2'].shape == self.x[['x1']].shape),'shape missmatch for neural network input')

        
        
        self.assertEqual(parsed_formula_content["loc"]['struct_shapes'], 1)
        self.assertEqual(parsed_formula_content["loc"]['P'].shape, (1, 1))
        self.assertTrue((parsed_formula_content["loc"]['P']==0).all())
        self.assertEqual(parsed_formula_content["scale"]['struct_shapes'], 2)
        self.assertEqual(parsed_formula_content["scale"]['P'].shape, (2, 2))
        self.assertTrue((parsed_formula_content["scale"]['P']==0).all())
        
        
        self.assertEqual(list(parsed_formula_content['loc']['deep_models_dict'].keys()), ['d1'])
        self.assertEqual(parsed_formula_content['loc']['deep_models_dict']['d1'],deep_models_dict['d1']['model'])
        self.assertEqual(parsed_formula_content['loc']['deep_shapes']['d1'], deep_models_dict['d1']['output_shape'])
        
        self.assertEqual(list(parsed_formula_content['scale']['deep_models_dict'].keys()), ['d2'])
        self.assertEqual(parsed_formula_content['scale']['deep_models_dict']['d2'],deep_models_dict['d2']['model'])
        self.assertEqual(parsed_formula_content['scale']['deep_shapes']['d2'], deep_models_dict['d2']['output_shape'])
        
        
    def test_smoothingspline_parse_formulas(self):
        """
        Test if parse_formulas is correctly dealing with smoothingsplines and computes the right P-matrix
        """
        # define formulas

        formulas = dict()
        formulas['loc'] = '~1'
        formulas['scale'] = '~1 + x1 + spline(x1,bs="bs",df=10, degree=3)'

        # define distributions and network names
        cur_distribution = 'Normal'
        family = Family(cur_distribution)

        deep_models_dict = dict()

        #call parse_formulas
        parsed_formula_content, meta_datadict = parse_formulas(family, formulas, self.x, deep_models_dict)       

        ground_truth_loc = dmatrix('~1', self.x, return_type='dataframe').to_numpy()
        ground_truth_scale = dmatrix('~1 + x1 + spline(x1,bs="bs",df=10, degree=3)', self.x, return_type='dataframe').to_numpy()
        #test if shapes of design matrices and P are as correct
        self.assertTrue((meta_datadict['loc']['structured'] == ground_truth_loc).all())
        self.assertTrue((meta_datadict['scale']['structured'] == ground_truth_scale).all())
        self.assertTrue((meta_datadict['loc']['structured'].shape == ground_truth_loc.shape),'shape missmatch')
        self.assertTrue((meta_datadict['scale']['structured'].shape == ground_truth_scale.shape),'shape missmatch')
        self.assertEqual(parsed_formula_content["loc"]['struct_shapes'], 1)
        self.assertEqual(parsed_formula_content["loc"]['P'].shape, (1, 1))
        self.assertTrue((parsed_formula_content["loc"]['P']==0).all())
        self.assertEqual(parsed_formula_content["scale"]['struct_shapes'], 12)
        self.assertEqual(parsed_formula_content["scale"]['P'].shape, (12, 12))
        self.assertTrue((parsed_formula_content["scale"]['P'][2:,2:]==spline(self.x.x1,bs="bs",df=10, degree=3,return_penalty = True)).all())

if __name__ == '__main__':
    unittest.main()