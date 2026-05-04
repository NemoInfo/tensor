benchmark_simd :: () {
  N :: 1024;
  I :: 100;
  total :: #run cast(f64)(N * N * I);

  a := range(.[N, N], f32); defer tensor_free(a);
  b := ones( .[N, N], f32); defer tensor_free(b);
  c := ones( .[N, N], f32); defer tensor_free(c);

  naive_time : f64 = 0.0;
  {
    start := current_time_monotonic();
    tmp := raw_tensor(.[N,N], f32); defer tensor_free(*tmp);
    res := raw_tensor(.[N,N], f32); defer tensor_free(*res);
    for 1..I {
      for 0..tmp.count-1 tmp.data[it] = a.data[it] * b.data[it];
      for 0..res.count-1 res.data[it] = tmp.data[it] + c.data[it];
    }
    naive_time = to_float64_seconds(current_time_monotonic() - start);
  }
  print("flat (naive): % Melem/s\n", total / naive_time  / 1_000_000);

  smart_time : f64 = 0.0;
  {
    start := current_time_monotonic();
    res := raw_tensor(.[N,N], f32); defer tensor_free(*res);
    ad := a.ptr;
    bd := b.ptr;
    cd := c.ptr;
    rd := res.ptr;
    for 1..I
      for 0..res.count-1 rd[it] = ad[it] * bd[it] + cd[it];
    smart_time = to_float64_seconds(current_time_monotonic() - start);
  }
  print("flat (optimal): % Melem/s\n", total / smart_time  / 1_000_000);

  fused_time : f64 = 0.0;
  {
    start := current_time_monotonic();
    ret   := tensor(shape_of(a * b + c), f32); defer tensor_free(ret);
    for 1..I compute(a * b - c, *ret);
    fused_time = to_float64_seconds(current_time_monotonic() - start);
  }
  print("fused: % Melem/s\n", total / fused_time  / 1_000_000);

  avx_time : f64 = 0.0;
  {
    start := current_time_monotonic();
    res := raw_tensor(.[N, N], f32); defer tensor_free(*res);
    for 1..I simd_muladd_avx(a.data.data, b.data.data, c.data.data, res.data.data, res.count);
    avx_time = to_float64_seconds(current_time_monotonic() - start);
  }
  print("avx (optimal): % Melem/s\n", total / avx_time  / 1_000_000);

  simd : f64 = 0.0;
  {
    start := current_time_monotonic();
    ret   := tensor(shape_of(a * b + c), f32); defer tensor_free(ret);
    for 1..I compute(a * b + c, *ret, simd = true);
  
    simd = to_float64_seconds(current_time_monotonic() - start);
  }
  print("avx fused: % Melem/s\n", total / simd  / 1_000_000);
}

benchmark_function :: (fn: Code) #expand {
  fn_name :: #run print_code(fn);
  time : f64 = 0.;
  start := current_time_monotonic();
  for 1..I #insert fn;
  dt := to_float64_seconds(current_time_monotonic() - start);
  println("%: %M elem/s", fn_name, TOTAL / dt / 1_000_000);
}

simd_muladd_avx :: (a: *f32, b: *f32, c: *f32, res: *f32, n: s64) {
    simd_n := n & ~(cast(s64) 7);

    for 0..simd_n / 8 - 1 {
      #asm AVX {
        va: vec;
        movups va,     [a + it * 8];
        mulps  va, va, [b + it * 8];
        addps  va, va, [c + it * 8];
        movups [res + it * 8], va;
      }
    }

    for simd_n..n - 1
        res[it] = a[it] * b[it] + c[it];
}

test :: () {
  N :: 10;
  a := range(.[N, N], f32); defer tensor_free(a);
  b := ones(.[N, N], f32);  defer tensor_free(b);
  c := ones(.[N, N], f32);  defer tensor_free(c);
  // res := ((a*(c + b) + a*b) * (a*b + a*b)) * ((a*b + a*b) * (c*a + a*b));
  res := a * b + c;
  #run dbg(type_of(res));
  // tensor_print(compute(res, simd = true));
}
