Tensor math library for JAI. The idea is to compute a type grapth of operations
at compile time that resolves to optimal vectorized assembly code.

## Usage guide

To run the benchmark 

```sh
jai benchmark.jai -quiet -o +Autorun
python3 sandbox/bench_numpy.py # Reference numpy implementation of the same operation
python3 sandbox/bench_torch.py # Reference torch implementation of the same operation
```

And bellow is a demo of the current API

```cpp
N := 10;
a := range(.[N, N], f32);  defer tensor_free(a);
b := ones (.[N, N], f32);  defer tensor_free(b);
c := ones (.[N, N], f32);  defer tensor_free(c);
ret := compute(a * b + c); defer tensor_free(ret);
tensor_print(ret);

for 1..10 a *= 2.;
tensor_print(a);
```
