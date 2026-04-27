benchmark :: () {
  N :: 1024;
  I :: 100;
  total :: #run cast(f64)(N * N * I);

  a := range(.[N, N], f32); defer tensor_free(a);
  b := ones( .[N, N], f32); defer tensor_free(b);
  c := ones( .[N, N], f32); defer tensor_free(c);

  fused_time : f64 = 0.0;
  {
    start := current_time_monotonic();
    for 1..I {
      res := compute(a * b + c);
      tensor_free(res);
    }
    fused_time = to_float64_seconds(current_time_monotonic() - start);
  }
  print("Fused a*b+c: % Melem/s\n", total / fused_time  / 1_000_000);

  fused_time_simd : f64 = 0.0;
  {
    start := current_time_monotonic();
    for 1..I {
      res := compute(a * b + c, simd = true);
      tensor_free(res);
    }
    fused_time_simd = to_float64_seconds(current_time_monotonic() - start);
  }
  print(" SIMD a*b+c: % Melem/s\n", total / fused_time_simd  / 1_000_000);

  naive_time : f64 = 0.0;
  {
    start := current_time_monotonic();
    for 1..I {
      tmp := raw_tensor(.[N,N], f32);
      for 0..tmp.count-1 tmp.data[it] = a.data[it] * b.data[it];
      res := raw_tensor(.[N,N], f32);
      for 0..res.count-1 res.data[it] = tmp.data[it] + c.data[it];
      array_free(tmp.data); array_free(tmp.shape); array_free(tmp.stride);
      array_free(res.data); array_free(res.shape); array_free(res.stride);
    }
    naive_time = to_float64_seconds(current_time_monotonic() - start);
  }
  print("Naive a*b+c: % Melem/s\n", total / naive_time  / 1_000_000);

  bcast_time : f64 = 0.0;
  {
    row := ones(.[1, N], f32); defer tensor_free(row);
    col := ones(.[N, 1], f32); defer tensor_free(col);
    start := current_time_monotonic();
    for 1..I {
      res := compute(row + col);
      tensor_free(res);
    }
    bcast_time = to_float64_seconds(current_time_monotonic() - start);
  }
  print("(1,N)+(N,1): % Melem/s\n", total / bcast_time / 1_000_000);
}

benchmark_simd :: () {
  N :: 1024;
  I :: 100;
  total :: #run cast(f64)(N * N * I);

  a := range(.[N, N], f32); defer tensor_free(a);
  b := ones( .[N, N], f32); defer tensor_free(b);
  c := ones( .[N, N], f32); defer tensor_free(c);

  fused_time : f64 = 0.0;
  {
    start := current_time_monotonic();
    for 1..I {
      res := compute(a * b - c, simd=true);
      tensor_free(res);
    }
    fused_time = to_float64_seconds(current_time_monotonic() - start);
  }
  print("Fused a*b+c: % Melem/s\n", total / fused_time  / 1_000_000);

  fused_time_simd : f64 = 0.0;
  {
    start := current_time_monotonic();
    for 1..I {
      res := compute(a * b + c, simd = true);
      tensor_free(res);
    }
    fused_time_simd = to_float64_seconds(current_time_monotonic() - start);
  }
  print(" SIMD a*b+c: % Melem/s\n", total / fused_time_simd  / 1_000_000);

  naive_time : f64 = 0.0;
  {
    start := current_time_monotonic();
    for 1..I {
      tmp := raw_tensor(.[N,N], f32);
      for 0..tmp.count-1 tmp.data[it] = a.data[it] * b.data[it];
      res := raw_tensor(.[N,N], f32);
      for 0..res.count-1 res.data[it] = tmp.data[it] + c.data[it];
      array_free(tmp.data); array_free(tmp.shape); array_free(tmp.stride);
      array_free(res.data); array_free(res.shape); array_free(res.stride);
    }
    naive_time = to_float64_seconds(current_time_monotonic() - start);
  }
  print("Naive a*b+c: % Melem/s\n", total / naive_time  / 1_000_000);
}


test :: () {
  N :: 4;
  a := range(.[N, N], f32); defer tensor_free(a);
  b := range(.[N, N], f32); defer tensor_free(b);
  res := compute(a + b, simd=true);
  tensor_print(res);
}
