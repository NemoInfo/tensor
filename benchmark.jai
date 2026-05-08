N :: 1024;

benchmark :: () {

  a := range(.[N, N], f32); defer tensor_free(a);
  b := ones( .[N, N], f32); defer tensor_free(b);
  c := ones( .[N, N], f32); defer tensor_free(c);
  expr0 := a * c + b;
  expr1 := (a*(c + b) + a*b);
  expr2 := (a+(c + b) + (a+b)) + ((a+b) + (b+c)); // this expression makes simd version poop it's pants
  ret := tensor(shape_of(expr1), f32); defer tensor_free(ret);

  // benchmark_function(naive(a, b, c, *ret));  
  // benchmark_function(fused(a, b, c, *ret)); // Reference for optimal asssembly for a * b + c (no simd)
  benchmark_function(compute(expr2, *ret));
  // benchmark_function(avx_optimal(a, b, c, *ret)); // Reference for optimal asssembly for a * b + c (simd)
  // benchmark_function(compute_simd(expr2, *ret));
  benchmark_function(manual_optimal_expr2(a, b, c, *ret)); // Reference for optimal asssembly for a * b + c (simd)
  benchmark_function(compute_simd(expr2, *ret)); // More complex instruction
}

manual_optimal_expr2 :: (a: Tensor(f32), b: Tensor(f32), c: Tensor(f32), ret: *Tensor(f32)) {
    a_ptr := a.ptr;
    b_ptr := b.ptr;
    c_ptr := c.ptr;
    ret_ptr := ret.ptr;
    simd_n := ret.count & ~(cast(s64) 7);

    for 0..simd_n/8 - 1 {
        i := it * 32;   // byte offset: 8 floats * 4 bytes each

        #asm AVX {
          ymm0: vec === 0;
          ymm1: vec === 1;
          ymm2: vec === 2;
          ymm3: vec === 3;
          ymm4: vec === 4;
          ymm5: vec === 5;
          ymm6: vec === 6;
          ymm7: vec === 7;
          ymm8: vec === 8;
          ymm9: vec === 9;

          // Load a, b, c
          movups.y ymm0, [a_ptr + i];
          movups.y ymm1, [b_ptr + i];
          movups.y ymm2, [c_ptr + i];

          // t1 = a + b
          addps.y  ymm3, ymm0, ymm1;   // ymm3 = a + b

          // t2 = c + b
          addps.y  ymm4, ymm2, ymm1;   // ymm4 = c + b

          // tmp = a + t2 = a + (c + b)
          addps.y  ymm5, ymm0, ymm4;   // ymm5 = a + (c + b)

          // sum1 = tmp + t1 = a + (c + b) + (a + b)
          addps.y  ymm6, ymm5, ymm3;

          // t3 = b + c
          addps.y  ymm7, ymm1, ymm2;   // ymm7 = b + c

          // sum2 = t1 + t3 = (a + b) + (b + c)
          addps.y  ymm8, ymm3, ymm7;

          // final = sum1 + sum2
          addps.y  ymm9, ymm6, ymm8;

          movups.y [ret_ptr + i], ymm9;
        }
    }

    // scalar fallback unchanged
    for simd_n..ret.count-1 {
        ret.data[it] = a.data[it] + (c.data[it] + b.data[it]) + (a.data[it] + b.data[it]) + (a.data[it] + b.data[it]) + (b.data[it] + c.data[it]);
    }
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
  res := tprint("% it/s (%ms elapsed)", iterations / dt, formatInt(cast (s64) (dt * 1000), digits_per_comma=3, comma_string = ","));
  println("+%+\n|%|\n+%+", fn_name, center_pad(res, 80), center_pad("", 80, #char "-"));
}

test :: () {
  N := 10;
  a := range(.[N, N], f32);  defer tensor_free(a);
  b := ones (.[N, N], f32);  defer tensor_free(b);
  c := ones (.[N, N], f32);  defer tensor_free(c);
  expr0 := a * c + b;
  expr1 := (a*(c + b) + a*b);
  expr2 := (a+(c + b) + (a+b)) + ((a+b) + (b+c)); // this expression makes simd version poop it's pants
  ret := compute_simd(expr2); defer tensor_free(ret);
  tensor_print(ret);

  // op-assign (*=) implicity does compute_simd(a * 2, *a);
  //  not sure if it's a good idea to hide the call to compute 
  //  but it sure gives Pythonic syntax =))
  for 1..10 a *= 2.;
  tensor_print(a);
  // tensor_print(b);
  // a += b;
  // tensor_print(a);

}

main :: () {
  // test();
  benchmark();
}

#load "tensor.jai";
#load "util.jai";
