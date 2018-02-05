import numpy as np
from Distance import Dist
import time

def kMeans(data, k=3, max_iter=100, tol=1e-20, diff_type='max', loss = Dist.Euclidean, print_epoch=True, name="kMeans"):
    """
    k: number of group
    max_iter: max allowed number of iteration
    tol: under this tolerance, regard the differences are the same
    diff_type:  'max': max_diff or 'mean': mean_diff is restricted within tol
    """
    st = time.time()
    N = data.shape[0]
    idx = np.random.choice(N,k,replace=False) 
    print ('\n{}: k={}, max_iter: {}, tol:{}, starting_points={}'   \
           .format(name, k, max_iter, tol, idx ))
    m = data[idx].copy()
    m_new = m.copy() #initial value
    it = 0 #initial
    C = None #initial
    boo = True
    while ( boo ):
        m = m_new.copy() 
        C = [set({}) for i in range(k)] # empty set for each class
        for j in range(N): # each sample
            x_j = data[j] 
            d_ji = [loss(x_j,m[i]) for i in range(k)] # distant to each class
            lambda_j = np.argmin(d_ji) # assign to which class
            C[lambda_j].add(j) # assign j to the class
        diff = []
        for i in range(k):
            _m_new = np.mean([data[j] for j in C[i]],axis=0)# new m for class i
            _diff = np.abs(_m_new - m[i])
            _diff_ = np.max(_diff) if diff_type=='max' else mean_diff
            diff.append(_diff_)
            if ( _diff_ > tol):
                m_new[i] = _m_new
            else:
                m_new[i] = m[i]
        
        if print_epoch: print ('Epoch: {},  C={}'.format(it, C))
        # avoid empty set
        from avoid_empty_set import avoid_empty_set
        C = avoid_empty_set(C)
        it+=1
        boo = it < max_iter and (m_new != m).any()
    print ("{}: clustering take {:.2f} sec".format(name, time.time()-st))
    return C

 