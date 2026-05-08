import numpy as np
from time import time

N = 1024 
I = 1000

a = np.arange(N * N, dtype=np.float32).reshape((N, N))
b = np.ones((N, N), dtype=np.float32)
c = np.ones((N, N), dtype=np.float32)

if __name__ == "__main__":
  start = time()
  for i in range(I): 
    r = (a+(c + b) + (a+b)) + ((a+b) + (b+c))

  dt = time() - start
  print(f"numpy: {I / dt:.6f} it/s ({int(dt * 1000):,}ms elapsed)");

