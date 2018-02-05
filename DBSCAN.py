import numpy as np
from Distance import Dist
import time

def find_neighbors(data, index, dist, epsilon):
    """
    find neighbors for data[i], return indexes
    """
    Nieghbor = set() 
    center_point = data[index]
    for i, d in enumerate(data):
        if i != index:
            if dist(d, center_point) < epsilon:
                Nieghbor.add(i)
    return Nieghbor
    

def DBSCAN(data, epsilon=0.001, MinPts =3, max_iter=100, tol=1e-20, diff_type='max', dist = Dist.Euclidean, print_epoch=False, name="DBSCAN"):
    """
    epsilon: samples distance not larger than
    MinPts: if within distance epsilon, there are at least MinPts samples, it's a core object
    max_iter: max allowed number of iteration
    tol: under this tolerance, regard the differences are the same
    diff_type:  'max': max_diff or 'mean': mean_diff is restricted within tol
    """
    st = time.time()
    N = data.shape[0]
    omega = set() # init core object set
    Nieghbor_all = [] # init Nieghbor for each points
    for index, d in enumerate(data):
        Nieghbor = find_neighbors(data, index, dist, epsilon)
        Nieghbor_all.append(Nieghbor)
        if len(Nieghbor) >= MinPts:
            omega.add(index)  
    print ("omega (core object set) = {}".format(omega))  
    print ("Nieghbor_all=\n0:{}\n1:{}\n2:{}\n3:{}\n4:{}\n5:{}\n6:{}\n7:{}\n8:{}\n9:{}\n10:{}\n11:{}\n12:{}\n13:{}\n14:{}\n".format(*Nieghbor_all))
            
    k = 0 # init number of clusters
    unvisit = set(range(N)) # init unvisit
    C = [] # init list of clusters
    while( len(omega) != 0 ):
        unvisit_old = unvisit.copy()
        idx = list(omega)[np.random.randint(len(omega))] # idx of selected_core
        if print_epoch: print ("\nidx of selected_core = {}".format(idx)) 
        Q = [ idx ] #init
        unvisit.remove(idx)
        while ( len(Q) != 0 ):
            q = Q[0]
            Q.remove(q)
            if print_epoch: print ("k= {}, q={}, Q={}, unvisit={}, Nieghbor={}".format(k,q,Q,unvisit, Nieghbor_all[q])) 
            if len(Nieghbor_all[q]) >= MinPts:
                delta = Nieghbor_all[q].intersection(unvisit)
                for e in delta:
                    Q.append(e)
                    unvisit.remove(e)
            if print_epoch: print ("k= {}, q={}, Q={}, unvisit={}".format(k,q,Q,unvisit))         
        k += 1
        c_k = unvisit_old.difference(unvisit) #create a cluster C_k
        if print_epoch: print ("unvisit_old= {}, unvisit={}, c_k={}".format(unvisit_old, unvisit, c_k))  
        C.append(c_k) 
        omega = omega.difference(c_k)
        if print_epoch: print ("omega= {}".format(omega))  
    return C
            
            
        
        
        
        
        
        
    
    
    
    