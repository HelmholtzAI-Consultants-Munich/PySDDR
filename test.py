import unittest

from utils import parse_formulas



class Testparse_formulas(unittest.TestCase):
    def test__shapes_in_parse_formulas(self):
        """
        Test if shapes of designmatrices and P returned my parse_formulas are correct
        """
        # load data
        iris = sm.datasets.get_rdataset('iris').data
        x = iris.rename(columns={'Sepal.Length':'x1','Sepal.Width':'x2','Petal.Length':'x3','Petal.Width':'x4','Species':'y'})
        # define formulas

        formulas = dict()
        formulas['loc'] = '~1 + x1 + x2 + spline(x1,bs="bs",df=10, degree=3)'
        formulas['scale'] = '~1'

        # define distributions and network names
        cur_distribution = 'normal'
        family = {'normal':{'loc': 'whateva', 'scale': 'whateva2'}, 'poisson': {'loc': 'whateva'}, 'binomial':{'n': 'whateva', 'p': 'whateva'}}

        deep_models_dict = dict()
        deep_models_dict['loc'] = dict()
        deep_models_dict['scale'] = dict()

        deep_shapes = dict()
        deep_shapes['loc'] = dict()
        deep_shapes['scale'] = dict()

        #call parse_formulas
        parsed_formula_content, meta_datadict = parse_formulas(family, formulas, x, cur_distribution, deep_models_dict, deep_shapes)       

        #test if shapes of design matrices and P are as correct
        self.assertEqual(meta_datadict['loc']['structured'].shape, (150, 13))
        self.assertEqual(meta_datadict["scale"]['structured'].shape, (150, 1))
        self.assertEqual(parsed_formula_content["loc"]['struct_shapes'], 13)
        self.assertEqual(parsed_formula_content["loc"]['P'].shape, (13, 13))
        self.assertEqual(parsed_formula_content["scale"]['struct_shapes'], 1)
        self.assertEqual(parsed_formula_content["scale"]['P'].shape, (1, 1))

if __name__ == '__main__':
    unittest.main()