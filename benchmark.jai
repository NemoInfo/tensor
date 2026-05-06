N :: 1024;

benchmark :: () {

  a := range(.[N, N], f32); defer tensor_free(a);
  b := ones( .[N, N], f32); defer tensor_free(b);
  c := ones( .[N, N], f32); defer tensor_free(c);
  expr := a * b + c;
  // expr := (a*(c + b) + a*b) * (a/b + b*c); this expression makes the simd version poop it's pants
  ret := tensor(shape_of(expr), f32); defer tensor_free(ret);

  benchmark_function(naive(a, b, c, *ret));  
  benchmark_function(fused(a, b, c, *ret)); // Reference for optimal asssembly for a * b + c (no simd)
  benchmark_function(compute(expr, *ret));
  benchmark_function(avx_optimal(a, b, c, *ret)); // Reference for optimal asssembly for a * b + c (simd)
  benchmark_function(compute_simd(expr, *ret));
  benchmark_function(compute_simd(a * (c + b) - a / b, *ret)); // More complex instruction
}


fused :: (a: Tensor(f32), b: Tensor(f32), c: Tensor(f32), ret: *Tensor(f32)) {
  for 0..ret.count-1 ret.data[it] = a.data[it] * b.data[it];
}

naive :: (a: Tensor(f32), b: Tensor(f32), c: Tensor(f32), ret: *Tensor(f32)) {
  for 0..ret.count-1 ret.data[it] = a.data[it] * b.data[it];
  for 0..ret.count-1 ret.data[it] = ret.data[it] + c.data[it];
}

avx_optimal :: (a: Tensor(f32), b: Tensor(f32), c: Tensor(f32), ret: *Tensor(f32)) {
  a_ptr := a.ptr;
  b_ptr := b.ptr;
  c_ptr := c.ptr;
  ret_ptr := ret.ptr;
  simd_n  := ret.count & ~(cast(s64) 7);

  for 0..simd_n / 8 - 1 {
    #asm AVX {
      va: vec;
      movups va,     [a_ptr + it * 8];
      mulps  va, va, [b_ptr + it * 8];
      addps  va, va, [c_ptr + it * 8];
      movups [ret_ptr + it * 8], va;
    }
  }

  for simd_n..ret.count - 1
    ret_ptr[it] = a_ptr[it] * b_ptr[it] + c_ptr[it];
}

benchmark_function :: (fn: Code, iterations := 1000) #expand {
  fn_name :: #run center_pad(print_code(fn), 80, #char "-");
  start := current_time_monotonic();
  for 1..iterations #insert fn;
  dt := to_float64_seconds(current_time_monotonic() - start);
  res := tprint("% it/s", iterations / dt);
  println("+%+\n|%|\n+%+", fn_name, center_pad(res, 80), center_pad("", 80, #char "-"));
}

test :: () {
  N := 10;
  a := range(.[N, N], f32);  defer tensor_free(a);
  b := ones (.[N, N], f32);  defer tensor_free(b);
  c := ones (.[N, N], f32);  defer tensor_free(c);
  ret := compute(a * b + c); defer tensor_free(ret);
  tensor_print(ret);

  // op-assign (*=) implicity does compute_simd(a * 2, *a);
  //  not sure if it's a good idea to hide the call to compute 
  //  but it sure gives Pythonic syntax =))
  for 1..10 a *= 2.;

  tensor_print(a);
}

main :: () {
  // test();
  benchmark();
}

#load "tensor.jai";
#load "util.jai";
