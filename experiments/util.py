def find_sub_list(sl, l):
    #'''find the first occurrence of a sublist in list: #from https://stackoverflow.com/a/17870684'''
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind : ind + sll] == sl:
            return ind, ind + sll - 1
