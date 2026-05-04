RawTensor :: struct (T: Type) {
  data:      []T;
  #overlay (data.count) count: s64;
  #overlay (data.data)  ptr: *T;
  shape:     []s64;
  stride:    []s64;
  #overlay (shape.count) ndim: s64;
  flags:     TensorFlags;
  allocator: Allocator;
}

Tensor :: struct($T: Type) { using tensor: *RawTensor(T);}

TensorFlags :: enum_flags u8 {
  CONTIGUOUS;
  WRITEABLE;
  OWNED;
}

compute_strides :: (shape: []s64) -> []s64 {
  stride := NewArray(shape.count, s64);
  stride[shape.count-1] = 1;
  for < 0..shape.count-2 
    stride[it] = stride[it+1] * shape[it+1];
  return stride;
}

count :: inline (shape: []s64) -> s64 {
  n := 1; for shape n *= it;
  return n;
}

raw_tensor :: (shape: []s64, $T: Type, allocator : Allocator = context.allocator) -> RawTensor(T) {
  push_allocator(allocator);
  t: RawTensor(T);
  t.shape     = array_copy(shape);
  t.stride    = compute_strides(shape);
  t.allocator = allocator;
  t.flags     = .CONTIGUOUS | .OWNED | .WRITEABLE;
  t.data      = NewArray(count(shape), T);
  return t;
}

tensor :: (shape: []s64, $T: Type, allocator : Allocator = context.allocator) -> Tensor(T) {
  res := New(RawTensor(T),,allocator);
  res.* = raw_tensor(shape, T, allocator);
  return .{ res };
}

tensor_free :: (t: *RawTensor($T)) {
  if t.flags & .OWNED array_free(t.data);
  array_free(t.shape);
  array_free(t.stride);
}

tensor_free :: (t: Tensor($T)) { tensor_free(t.tensor); }

element :: (el: $T, shape: []s64, allocator : Allocator = context.allocator) -> Tensor(T) {
  res := tensor(shape, T, allocator);
  for 0..res.count-1 res.data[it] = el;
  return res;
}

zeros :: (shape: []s64, $T: Type = f32, allocator : Allocator = context.allocator) -> Tensor(T) {
  return element(cast(T) 0, shape, allocator);
}

ones :: (shape: []s64, $T: Type = f32, allocator : Allocator = context.allocator) -> Tensor(T) {
  return element(cast(T) 1, shape, allocator);
}

range :: (shape: []s64, $T: Type = f32, allocator : Allocator = context.allocator) -> Tensor(T) {
  assert(cast(f64) Sloppy_Math.max_of_type(T) >= cast(f64) (count(shape) - 1),
    "Range maximum value (%) exceeds % upper bound (%).", (count(shape) - 1), T , Sloppy_Math.max_of_type(T));
  res := tensor(shape, T, allocator);
  for 0..res.count-1 res.data[it] = cast(T) it;
  return res;
}

shape_eq :: (a: RawTensor($T), b: RawTensor(T)) -> bool {
  if a.ndim != b.ndim return false;
  for a.shape if it != b.shape[it_index] return false;
  return true;
}

shape_eq :: (a: []s64, b: []s64) -> bool {
  if a.count != b.count return false;
  for a if it != b[it_index] return false;
  return true;
}

main :: () { test(); benchmark_simd(); }

tensor_print :: (a: Tensor($T)) { tensor_print(a.tensor); }
tensor_print :: (a: RawTensor($T)) {
  push_allocator(Default_Allocator.allocator);
  if a.ndim == 0  { print("%\n", a.data[0]); return; }

  print_dim :: (data: []T, shape: []s64, stride: []s64, dim: int, offset: s64, max_width: s64) {
    print("[");
    for 0..shape[dim]-1 {
      if dim == shape.count - 1 {
        if it > 0 print (" ");
        s := sprint("%", data[offset + it * stride[dim]]);
        for 0..max_width-s.count-1 print(" ");
        print("%", s);
        free(s);
      } else {
        if it > 0 {
          print(",");
          for 0..shape.count-dim-2 print("\n");
          for 0..dim print(" ");
        }
        print_dim(data, shape, stride, dim + 1, offset + it * stride[dim], max_width);
      }
    }
    print("]");
  }

  max_width := 0;
  for 0..a.count-1 {
    w := sprint("%", a.data[it]);
    if w.count > max_width  max_width = w.count;
    free(w);
  }

  print_dim(a.data, a.shape, a.stride, 0, 0, max_width);
  print("\n");
}

#import "Basic";
#load "tree.jai";
#load "util.jai";
#load "simd.jai";
#load "benchmark.jai";
Default_Allocator :: #import "Default_Allocator";
Sloppy_Math       :: #import "Sloppy_Math";
