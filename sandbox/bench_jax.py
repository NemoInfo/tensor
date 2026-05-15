import jax
import jax.numpy as jnp
from time import time

N = 1024
I = 1000

a = jnp.arange(N * N, dtype=jnp.float32).reshape((N, N))
b = jnp.ones((N,), dtype=jnp.float32)
c = jnp.ones((N, N), dtype=jnp.float32)

@jax.jit
def compute_r(a, b, c):
    return (a*(c + b) + a*b) * (a/b + b*c);

if __name__ == "__main__":
    _ = compute_r(a, b, c).block_until_ready()

    start = time()
    for _ in range(I):
        r = compute_r(a, b, c)
        r.block_until_ready()
    dt = time() - start

    print(f"jax: {I / dt:.6f} it/s ({int(dt * 1000):,} ms elapsed)")
