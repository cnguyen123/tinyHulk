import numpy as np
a = {}

#print(a)


def type_change(obj):
    if isinstance(obj, np.int32):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.float):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj



for k in a.keys():
    print(k)
    b = a.get(k)
    len_b = len(b)
    print(len_b)
    for i in range(0, len_b):
        for key in b[i]:
                a.get(k)[i][key] = type_change(a.get(k)[i].get(key))





