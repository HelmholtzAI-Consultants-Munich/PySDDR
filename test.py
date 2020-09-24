import unittest

from patsy import dmatrix
import statsmodels.api as sm
from utils import parse_formulas, spline
import numpy as np
from torch import nn

class Testparse_formulas(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super(Testparse_formulas, self).__init__(*args,**kwargs)
        # load data
        iris = sm.datasets.get_rdataset('iris').data
        self.x = iris.rename(columns={'Sepal.Length':'x1','Sepal.Width':'x2','Petal.Length':'x3','Petal.Width':'x4','Species':'y'})
        
    
    def test_shapes_in_parse_formulas(self):
        """
        Test if shapes of designmatrices and P returned my parse_formulas are correct
        """

        # define formulas

        formulas = dict()
        formulas['loc'] = '~1 + x1 + x2 + spline(x1,bs="bs",df=10, degree=3)'
        formulas['scale'] = '~1'

        # define distributions and network names
        cur_distribution = 'normal'
        dummy_family = {'normal':{'loc': 0, 'scale': 0}}

        deep_models_dict = dict()
        deep_models_dict['loc'] = dict()
        deep_models_dict['scale'] = dict()

        deep_shapes = dict()
        deep_shapes['loc'] = dict()
        deep_shapes['scale'] = dict()

        #call parse_formulas
        parsed_formula_content, meta_datadict = parse_formulas(dummy_family, formulas, self.x, cur_distribution, deep_models_dict, deep_shapes)       

        #test if shapes of design matrices and P are as correct
        self.assertEqual(meta_datadict['loc']['structured'].shape, (150, 13))
        self.assertEqual(meta_datadict["scale"]['structured'].shape, (150, 1))
        self.assertEqual(parsed_formula_content["loc"]['struct_shapes'], 13)
        self.assertEqual(parsed_formula_content["loc"]['P'].shape, (13, 13))
        self.assertEqual(parsed_formula_content["scale"]['struct_shapes'], 1)
        self.assertEqual(parsed_formula_content["scale"]['P'].shape, (1, 1))
        
    def test_patsyfreedummytest_parse_formulas(self):
        """
        Test if parse_formulas without assuming pasty is correct
        """
        # define formulas

        formulas = dict()
        formulas['loc'] = '~1'
        formulas['scale'] = '~1'

        # define distributions and network names
        cur_distribution = 'normal'
        dummy_family = {'normal':{'loc': 0, 'scale': 0}}

        deep_models_dict = dict()
        deep_models_dict['loc'] = dict()
        deep_models_dict['scale'] = dict()

        deep_shapes = dict()
        deep_shapes['loc'] = dict()
        deep_shapes['scale'] = dict()

        #call parse_formulas
        parsed_formula_content, meta_datadict = parse_formulas(dummy_family, formulas, self.x, cur_distribution, deep_models_dict, deep_shapes)       
        
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
        cur_distribution = 'normal'
        dummy_family = {'normal':{'loc': 0, 'scale': 0}}

        deep_models_dict = dict()
        deep_models_dict['loc'] = dict()
        deep_models_dict['scale'] = dict()

        deep_shapes = dict()
        deep_shapes['loc'] = {}
        deep_shapes['scale'] = dict()

        #call parse_formulas
        parsed_formula_content, meta_datadict = parse_formulas(dummy_family, formulas, self.x, cur_distribution, deep_models_dict, deep_shapes)       
        
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
        formulas['loc'] = '~1 + d1(x2)'
        formulas['scale'] = '~1 + x1'

        # define distributions and network names
        cur_distribution = 'normal'
        dummy_family = {'normal':{'loc': 0, 'scale': 0}}

        deep_models_dict = dict()
        deep_models_dict['loc'] = {'d1': nn.Sequential(nn.Linear(1,15))}
        deep_models_dict['scale'] = dict()

        deep_shapes = dict()
        deep_shapes['loc'] = {'d1': 42}
        deep_shapes['scale'] = dict()

        #call parse_formulas
        parsed_formula_content, meta_datadict = parse_formulas(dummy_family, formulas, self.x, cur_distribution, deep_models_dict, deep_shapes)       

        ground_truth_loc = dmatrix('~1', self.x, return_type='dataframe').to_numpy()
        ground_truth_scale = dmatrix('~1 + x1', self.x, return_type='dataframe').to_numpy()
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
        
    def test_smoothingspline_parse_formulas(self):
        """
        Test if parse_formulas is correctly dealing with smoothingsplines and computes the right P-matrix
        """
        # define formulas

        formulas = dict()
        formulas['loc'] = '~1 + d1(x2)'
        formulas['scale'] = '~1 + x1 + spline(x1,bs="bs",df=10, degree=3)'

        # define distributions and network names
        cur_distribution = 'normal'
        dummy_family = {'normal':{'loc': 0, 'scale': 0}}

        deep_models_dict = dict()
        deep_models_dict['loc'] = {'d1': nn.Sequential(nn.Linear(1,15))}
        deep_models_dict['scale'] = dict()

        deep_shapes = dict()
        deep_shapes['loc'] = {'d1': 42}
        deep_shapes['scale'] = dict()

        #call parse_formulas
        parsed_formula_content, meta_datadict = parse_formulas(dummy_family, formulas, self.x, cur_distribution, deep_models_dict, deep_shapes)       

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