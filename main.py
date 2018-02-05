from Dataset import load_data, get_C_ref
from Distance import Dist
from kMeans import kMeans
from DBSCAN import DBSCAN
from External_Index import ExternalIndex
from Internal_Index import InternalIndex
from Plot import plot


def __main__(algo='DBSCAN'):

    # data
    d_data, d_target, k_ = load_data()

    #cluster
    if algo=='kMeans':
        C_cluster = kMeans(d_data, 
                           k=3, 
                           loss=Dist.Euclidean, 
                           print_epoch=True) #kMeans
    elif algo=='DBSCAN':
        C_cluster = DBSCAN(d_data, 
                           epsilon=0.2, 
                           MinPts=1, 
                           dist=Dist.Euclidean, 
                           print_epoch=True) #DBSCAN
    else:
        print("Algorithm not support!"); return
    print('C_cluster=', C_cluster)

    # External Index
    if algo=='DBSCAN':
        C_ref = get_C_ref(C_cluster, d_target, k_)
        print('C_ref=',C_ref)
        exi = ExternalIndex(d_data, C_cluster, C_ref)
        exi.__print__()

    # Internal Index
    ini = InternalIndex(d_data, C_cluster)
    ini.__print__()

    #plot
    plot(C_cluster, d_data)


__main__()