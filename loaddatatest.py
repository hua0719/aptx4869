import numpy as np

x = np.load('ISLES/T1.npz')['arr_0']
x = x[:, :150, 0:-6, 34:-36]
