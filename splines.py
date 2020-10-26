from patsy.state import stateful_transform
import statsmodels.api as sm
from statsmodels.gam.api import CyclicCubicSplines, BSplines
from patsy.util import have_pandas, no_pickling, assert_no_pickling
import numpy as np

class Spline(object):
    """
     Class for computation of spline basis functions and and smooting penalty matrix for differents types of splines (BSplines, Cyclic cubic splines).
     Compatible with patsy statefull transform.
    
     Parameters
     ----------
         x: Pandas.DataFrame
             A data frame holding all the data 
         bs: string, default is 'bs'
             The type of splines to use - default is b splines, but can also use cyclic cubic splines if bs='cc'
         df: int, default is 4
             Number of degrees of freedom (equals the number of columns in s.basis)
         degree: int, default is 3
             degree of polynomial e.g. 3 -> cubic, 2-> quadratic
         return_penalty: bool, default is False
             has no function - necessary for backwards compatibility with the tests. Should be cleaned up at some point.
     Returns
     -------
         The function returns one of:
         s.basis: The basis functions of the spline
         s.penalty_matrices: The penalty matrices of the splines 
     """
    def __init__(self):
        pass

    def memorize_chunk(self, x, bs, df=4, degree=3, return_penalty = False):
        assert bs == "bs" or bs == "cc", "Spline basis not defined!"
        if bs == "bs":
            self.s = BSplines(x, df=[df], degree=[degree], include_intercept=True)
        elif bs == "cc":
            self.s = CyclicCubicSplines(x, df=[df])
        
        self.penalty_matrices = self.s.penalty_matrices

    def memorize_finish(self):
        pass


    def transform(self, x, bs, df=4, degree=3, return_penalty = False):
        
        return self.s.transform(np.expand_dims(x.to_numpy(),axis=1)) 
            

    __getstate__ = no_pickling

spline = stateful_transform(Spline) #conversion of Spline class to patsy statefull transform
