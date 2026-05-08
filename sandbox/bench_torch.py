import torch
from time import time

N = 1024 
I = 1000

torch.set_num_threads(16) 
a = torch.arange(N * N, dtype=torch.float32).reshape((N, N))
b = torch.ones((N, N), dtype=torch.float32)
c = torch.ones((N, N), dtype=torch.float32)

if __name__ == "__main__":
  start = time()
  with torch.no_grad():
    for i in range(I): 
      r = (a+(c + b) + (a+b)) + ((a+b) + (b+c))

  dt = time() - start
  print(f"numpy: {I / dt:.6f} it/s ({int(dt * 1000):,}ms elapsed)");
