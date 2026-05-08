# Features
- [ ] Boolean operators, and more generally think about support for operations
        that do not return the same types as the operands
- [ ] Resolve `compute` vs `compute_simd` dichotomy
- [ ] Tie AVX/FMA functions to CPU capabilities
- [ ] Add SIMD support for `dtype`s other than `f32`

# Bugs
- [ ] For some reason benchmark is faster with +Autorun vs 
      just running the binary, either the cache is warmer or there is some funny bussiness
- [x] Debug performance hit in complex expressions
