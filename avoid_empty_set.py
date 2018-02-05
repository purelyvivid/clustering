
def avoid_empty_set(C):
    if set() in C:
        len_C = [len(s) for s in C]
        idx_empty = len_C.index(0)
        maxlen = max(len_C)
        idx_maxlen = len_C.index(maxlen)
        c_empty, c_maxlen = C[idx_empty], C[maxlen]
        print("avoid empty set: from {} and {}".format(c_empty, c_maxlen))
        c_empty.add(c_maxlen.pop())
        print("avoid empty set: to {} and {}".format(c_empty, c_maxlen))
    return C