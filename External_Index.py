import numpy as np
from Distance import Dist
# external index
class ExtIn_fns: 
    
    fns_list = [ 'Jaccard', 'FM', 'RI' ]
    def build_L_from_C(C_set_list, m):
        assert len(C_set_list) <= m
        L_list = list(range(m))
        for Label, C_set in enumerate(C_set_list): 
            for i in C_set:
                L_list[i] = Label  
        return L_list

    def cal_abcd(D, C_cluster, C_ref):
        m = D.shape[0]
        #k, s = len(C_cluster), len(C_ref)
        L_cluster, L_ref = ExtIn_fns.build_L_from_C(C_cluster,m), ExtIn_fns.build_L_from_C(C_ref,m)
        a, b, c, d = 0, 0, 0, 0
        for i in range(m):
            #x_i = D[i]
            l_i_cls = L_cluster[i]
            l_i_ref = L_ref[i]
            for j in range(i+1, m):
                #x_j = D[j]
                l_j_cls = L_cluster[j]
                l_j_ref = L_ref[j]
                if l_i_cls == l_j_cls:
                    if l_i_ref == l_j_ref:
                        a += 1
                    else: 
                        b += 1
                else:
                    if l_i_ref == l_j_ref:
                        c += 1
                    else: 
                        d += 1
        assert a+b+c+d == m*(m-1)/2
        return a,b,c,d

    def Jaccard(a,b,c,d):
        assert a+b+c > 0
        return float(a)/(a+b+c)

    def FM(a,b,c,d):
        assert a+b > 0 and a+c > 0
        return np.sqrt(float(a)/(a+b)*float(a)/(a+c)) 

    def RI(a,b,c,d):
        assert a+b+c+d > 0
        return 2*float(a+d)/((a+b+c+d)*2) # (a+b+c+d)*2 == m*(m-1)

class ExternalIndex():
    
    score = {}
    fns = ExtIn_fns
    
    def __init__(self, D, C_cluster, C_ref, index='all'):
        """
        D = {x_1, x_2, ..., x_m}
        C_cluster = [ C_1 = set({...}), ..., C_k = set({...}) ]
        C_ref = [ C_1* = set({...}), ..., C_s* = set({...}) ]
        L_cluster = [ l_1 , l_2 , ...., l_m ]
        L_ref = [ l_1 , l_2 , ...., l_m ]
        """
        if hasattr(self.fns, index):
            fn = getattr(self.fns, index)
            score = self.cal_score(D, C_cluster, C_ref, fn)
            self.score.update({ index : score })
        elif index=='all':
            for index_name in self.fns.fns_list:
                fn = getattr(self.fns, index_name)
                score = self.cal_score(D, C_cluster, C_ref, fn)
                self.score.update({ index_name : score })                
        else:
            print("No this fn [%s] !!" % index)
        return
            
    def cal_score(self, D, C_cluster, C_ref, fn):
        a,b,c,d = self.fns.cal_abcd(D, C_cluster, C_ref)
        return fn(a,b,c,d) 
        
    def __print__(self):
        print("\nExternal Index:")
        for index_name in self.fns.fns_list:
            print("{} score = {:.2f}".format(index_name, self.score[index_name]))