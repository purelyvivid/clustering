import matplotlib.pyplot as plt

def plot(C_cluster, d_data):
    c_ = ['r', 'g', 'b']
    l_ = ['group_{}'.format(i+1) for i in range(3)]
    for i,g in enumerate(C_cluster):
        plt.scatter(*zip(*d_data[list(g)]), c=c_[i],label=l_[i])
        plt.legend(loc='lower right')
    plt.show()