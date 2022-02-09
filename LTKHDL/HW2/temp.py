import numpy as np


raw_ratings = np.genfromtxt("E:\D\hoc\NMKHDL\LTKHDL\HW2\u.data",dtype = np.uint64)
n_rows = raw_ratings.shape[0]
n_cols = raw_ratings.shape[1]

ratings = np.zeros((len(set(raw_ratings[:,0])),len(set(raw_ratings[:,1]))))
ratings[:] = np.nan

temp = raw_ratings.copy()

temp[:,0] -=1
temp[:,1] -=1

ratings[temp[:,0],temp[:,1]] = temp[:,2]


batch_size = 32
start = 0
end = batch_size

def f(x,y):
    a = np.abs(ratings[y]-ratings[x])
    a = np.nanmean(a)
    # nếu không có phim nào rate trùng nhau thì trả về 0
    if np.isnan(a):
        return 0
    w = 1/(a+0.001)
    return w

similarities = np.fromfunction(np.vectorize(f),(batch_size,len(ratings)),dtype='int')
