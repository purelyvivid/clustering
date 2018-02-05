import numpy as np
from Distance import Dist
# internal index
class IntIn_fns: 
    
    fns_list = [ 'DBI', 'DI' ]
    def avgC(D, C, dist_fn):
        """
        the avg dist in cluster C
        C = set({ ... })
        """
        assert type(D) == np.ndarray or type(D) == list
        #m = D.shape[0]
        assert type(C) == set or type(C) == list 
        n = len(C)
        assert n > 1
        C_list = list(C) # 'set' to 'list' for iteration
        sum_dist = 0
        for i in range(n):
            for j in range(i+1,n):
                #print(i, j )
                #print(C_list[i],C_list[j] )
                #print(D[C_list[i]],D[C_list[j]] )
                sum_dist += dist_fn(D[C_list[i]],D[C_list[j]])  
        return float(2)/n/(n-1)*sum_dist
    
    def diamC(D, C, dist_fn):  
        """
        the max dist in cluster C
        C = set({ ... })
        """
        assert type(D) == np.ndarray or type(D) == list
        #m = D.shape[0]
        assert type(C) == set or type(C) == list
        n = len(C)
        assert n > 1
        C_list = list(C) # 'set' to 'list' for iteration
        max_dist = 0
        for i in range(n):
            for j in range(i+1,n):
                dist = dist_fn(D[C_list[i]],D[C_list[j]]) 
                if (dist>max_dist):
                    max_dist = dist  
        return max_dist 
    
    def d_min(D, C_i, C_j, dist_fn):  
        """
        the min dist between cluster C_i and C_j
        C_i = set({ ... })
        """
        assert type(D) == np.ndarray or type(D) == list
        #m = D.shape[0]
        assert type(C_i) == set or type(C_i) == list
        assert type(C_j) == set or type(C_j) == list
        n_i, n_j = len(C_i) , len(C_j)
        C_i_list, C_j_list = list(C_i), list(C_j) # 'set' to 'list' for iteration
        min_dist = np.inf
        for i in range(n_i):
            for j in range(n_j):
                dist = dist_fn(D[C_i_list[i]],D[C_j_list[j]]) 
                if (dist<min_dist):
                    min_dist = dist  
        return min_dist 
    
    def d_cen(D, C_i, C_j, dist_fn):  
        """
        the dist between mean of cluster C_i and mean ofC_j
        C_i = set({ ... })
        """
        assert type(D) == np.ndarray or type(D) == list
        #m = D.shape[0]
        assert type(C_i) == set or type(C_i) == list
        assert type(C_j) == set or type(C_j) == list
        #n_i, n_j = len(C_i) , len(C_j)
        C_i_list, C_j_list = list(C_i), list(C_j) # 'set' to 'list' 
        #print(C_i_list, C_j_list)
        mu_i = np.mean( D[C_i_list], axis=0)
        mu_j = np.mean( D[C_j_list], axis=0)
        #print(mu_i, mu_j)
        return dist_fn(mu_i, mu_j)   

    def DBI(D, C_cluster, dist_fn): # smaller is better
        m = D.shape[0]
        k = len(C_cluster)
        assert k > 0 and m >= k
        sum_max = 0
        for i in range(k):
            max_ = 0
            for j in range(k): 
                if  j != i: 
                    a = IntIn_fns.avgC(D, C_cluster[i], dist_fn)
                    b = IntIn_fns.avgC(D, C_cluster[j], dist_fn)
                    c = IntIn_fns.d_cen(D, C_cluster[i], C_cluster[j], dist_fn)
                    assert c > 0
                    ind = (a+b)/c
                    if  ind > max_:
                        max_ = ind
            sum_max += max_
        return float(sum_max)/float(k) 
                    
    def DI(D, C_cluster, dist_fn): # larger is better
        m = D.shape[0]
        k = len(C_cluster)
        assert k > 0 and m >= k
        min_ = np.inf 
        for i in range(k):
            _min_ = np.inf
            for j in range(k): 
                if  j != i: 
                    a = IntIn_fns.d_min(D, C_cluster[i], C_cluster[j], dist_fn)
                    _max_ = 0
                    for l in range(k):
                        v = IntIn_fns.diamC(D, C_cluster[l], dist_fn)
                        if v > _max_:
                            _max_ = v
                    assert _max_ > 0
                    vv = float(a)/float(_max_)
                    if vv < _min_:
                        _min_ = vv
            assert _min_ < np.inf
            if _min_ < min_ :
                min_ = _min_
        assert min_ < np.inf
        return float(min_)
    
class InternalIndex():
    
    score = {}
    fns = IntIn_fns
    
    def __init__(self, D, C_cluster, dist_fn=Dist.Euclidean, index='all'):
        """
        D = {x_1, x_2, ..., x_m}
        C_cluster = [ C_1 = set({...}), ..., C_k = set({...}) ]
        L_cluster = [ l_1 , l_2 , ...., l_m ]
        """
        if hasattr(self.fns, index):
            fn = getattr(self.fns, index)
            score = self.cal_score(D, C_cluster, dist_fn, fn)
            self.score.update({ index : score })
        elif index=='all':
            for index_name in self.fns.fns_list:
                fn = getattr(self.fns, index_name)
                score = self.cal_score(D, C_cluster, dist_fn, fn)
                self.score.update({ index_name : score })                
        else:
            print("No this fn [%s] !!" % index)
        return
            
    def cal_score(self, D, C_cluster, dist_fn, fn):
        return fn(D, C_cluster, dist_fn)
        
    def __print__(self):
        print("\nInternal Index:")
        for index_name in self.fns.fns_list:
            print("{} score = {:.2f}".format(index_name, self.score[index_name]))            
    