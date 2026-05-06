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
      r = a * c + b

  print(f"torch: {I / (time() - start):.6f} it/s");

