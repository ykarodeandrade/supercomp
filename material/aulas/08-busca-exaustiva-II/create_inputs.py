import numpy as np
import numpy.random

N = int(input())
W = int(input())
print(N, W)
for i in range(N):
    w, v = np.random.rand(2)
    w = w * W + 1
    v *= 100
    print(int(w), int(v))