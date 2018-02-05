import numpy as np
# distant fn
class Dist:   
    def Minkowski(x1, x2, p=2):
        """
        x1, x2 can be a scalar or a matrix
        p is an integer
        return a scalar
        """
        return np.sum(np.abs(x1-x2)**p)**float(1/p)
        
    def Euclidean(x1, x2):
        return Dist.Minkowski(x1, x2, p=2)
    
    def Manhattan(x1, x2):
        return Dist.Minkowski(x1, x2, p=1)