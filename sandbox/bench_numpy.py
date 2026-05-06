import numpy as np
from time import time

N = 1024 
I = 100

a = np.arange(N * N, dtype=np.float32).reshape((N, N))
b = np.ones((N, N), dtype=np.float32)
c = np.ones((N, N), dtype=np.float32)

if __name__ == "__main__":
  start = time()
  for i in range(I): 
    r = (a*(c + b) + a*b)

  print(f"numpy: {I / (time() - start):.6f} it/s");

