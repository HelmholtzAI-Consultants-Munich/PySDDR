import unittest

import numpy as np
from torch import nn
import pandas as pd

import unittest

from dataset import SddrDataset

from patsy import dmatrix
import statsmodels.api as sm
from utils import parse_formulas, spline, Family


class TestSddrDataset(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super(TestSddrDataset, self).__init__(*args,**kwargs)
        
        self.current_distribution  = 'Poisson' #'Normal'

        self.formulas = {'rate': '~1 + x1 + x2 + spline(x1, bs="bs",df=9)+spline(x2, bs="bs",df=9)+d1(x1)+d2(x2)'}
        self.deep_models_dict = {
        'd1': {
            'model': nn.Sequential(nn.Linear(1,15)),
            'output_shape': 15},
        'd2': {
            'model': nn.Sequential(nn.Linear(1,3),nn.ReLU(), nn.Linear(3,8)),
            'output_shape': 8}
        }

        self.train_parameters = {
        'batch_size': 1000,
        'epochs': 2500,
        'regularization_params': {'rate': 1} #{'loc':1, 'scale':1}
        }

        self.family = Family(self.current_distribution)
        
        self.data_path = './test_data/x.csv'
        self.ground_truth_path = './test_data/y.csv'
        
        self.true_feature_names = ["x1","x2","x3","x4"]
        
        self.data = pd.read_csv(self.data_path ,sep=None,engine='python')
        self.target = pd.read_csv(self.ground_truth_path)
        
        self.true_x2_11 = np.float32(self.data.x2[11])
        self.true_target_11 = self.target.y[11]


    def test_pandasinput(self):
        """
        Test if SddrDataset correctly works with a pandas dataframe as input
        """
        
              
        data = pd.concat([self.data, self.target], axis=1, sort=False)

        dataset = SddrDataset(data = data, 
                                target = "y",
                                family = self.family,
                                formulas=self.formulas,
                                deep_models_dict=self.deep_models_dict)

        feature_names = dataset.get_list_of_feature_names()
        feature_test_value = dataset.get_feature('x2')[11]
        linear_input_test_value = dataset[11]["meta_datadict"]["rate"]["structured"].numpy()[2]
        deep_input_test_value = dataset[11]["meta_datadict"]["rate"]["d2"].numpy()[0]
        target_test_value = dataset[11]["target"].numpy()

        #test if outputs are equal to the true values in the iris dataset
        self.assertEqual(feature_names, self.true_feature_names)
        
        self.assertAlmostEqual(feature_test_value, self.true_x2_11,places=4)
        self.assertAlmostEqual(linear_input_test_value, self.true_x2_11,places=4)
        self.assertAlmostEqual(deep_input_test_value, self.true_x2_11,places=4)
        self.assertAlmostEqual(target_test_value, self.true_target_11,places=4)
      
    def test_pandasinputpandastarget(self):
        """
        Test if SddrDataset correctly works with a pandas dataframe as input and target also given as dataframe
        """
        
        dataset = SddrDataset(data = self.data, 
                                target = self.target,
                                family = self.family,
                                formulas=self.formulas,
                                deep_models_dict=self.deep_models_dict)

        feature_names = dataset.get_list_of_feature_names()
        feature_test_value = dataset.get_feature('x2')[11]
        linear_input_test_value = dataset[11]["meta_datadict"]["rate"]["structured"].numpy()[2]
        deep_input_test_value = dataset[11]["meta_datadict"]["rate"]["d2"].numpy()[0]
        target_test_value = dataset[11]["target"].numpy()

        #test if outputs are equal to the true values in the iris dataset
        self.assertEqual(feature_names, self.true_feature_names)
        
        self.assertAlmostEqual(feature_test_value, self.true_x2_11,places=4)
        self.assertAlmostEqual(linear_input_test_value, self.true_x2_11,places=4)
        self.assertAlmostEqual(deep_input_test_value, self.true_x2_11,places=4)
        self.assertAlmostEqual(target_test_value, self.true_target_11,places=4)
    
    
    def test_filepathinput(self):
        """
        Test if SddrDataset correctly works with file paths as inputs
        """
        dataset = SddrDataset(self.data_path, 
                        self.ground_truth_path,
                        self.family,
                        self.formulas,
                        self.deep_models_dict)
        
        feature_names = dataset.get_list_of_feature_names()
        feature_test_value = dataset.get_feature('x2')[11]
        linear_input_test_value = dataset[11]["meta_datadict"]["rate"]["structured"].numpy()[2]
        deep_input_test_value = dataset[11]["meta_datadict"]["rate"]["d2"].numpy()[0]
        target_test_value = dataset[11]["target"].numpy()

        #test if outputs are equal to the true values in the iris dataset
        self.assertEqual(feature_names, self.true_feature_names)
        
        self.assertAlmostEqual(feature_test_value, self.true_x2_11,places=4)
        self.assertAlmostEqual(linear_input_test_value, self.true_x2_11,places=4)
        self.assertAlmostEqual(deep_input_test_value, self.true_x2_11,places=4)
        self.assertAlmostEqual(target_test_value, self.true_target_11,places=4)


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
        parsed_formula_content, meta_datadict, dm_info_dict = parse_formulas(family, formulas, self.x, deep_models_dict)       
        
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
        parsed_formula_content, meta_datadict, dm_info_dict = parse_formulas(family, formulas, self.x, deep_models_dict)       
        
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
        parsed_formula_content, meta_datadict, dm_info_dict = parse_formulas(family, formulas, self.x, deep_models_dict)       

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
        parsed_formula_content, meta_datadict, dm_info_dict = parse_formulas(family, formulas, self.x, deep_models_dict)       

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