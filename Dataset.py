from sklearn.datasets import load_iris
import numpy as np

def load_data():
    d = load_iris()
    #print(d.data.shape)
    #print(d.target)
    k = len(np.unique(d.target))
    assert k==3

    # Sample Data
    idx = list(range(5))+list(range(50,55))+list(range(100,105))
    d_data = d.data[idx,2:] # only select 15 samples and two features 
    d_target = d.target[idx]
    #print(d_targe) # choose 5 from each class 
    
    return d_data, d_target, k

def get_C_ref(C_cluster, d_target, k):
    default_C = np.array_split(range(15),3)
    C_ref = []
    for i in range(k):
        c = np.zeros(shape=k)
        for j in C_cluster[i]:
            c[d_target[j]] += 1
        L = np.argmax(c)
        C_ref.append(set(default_C[L]) )
        return C_ref