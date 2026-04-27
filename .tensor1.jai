Tensor :: struct (T: Type) {
  data:      []T;
  #overlay (data.count) count: s64;
  shape:     []s64;
  stride:    []s64;
  #overlay (shape.count) ndim: s64;
  flags:     TensorFlags;
  allocator: Allocator;
}

count :: inline (shape: []s64) -> s64 {
  n := 1; for shape n *= it;
  return n;
}

TensorFlags :: enum_flags u8 {
  CONTIGUOUS;
  WRITEABLE;
  OWNED;
}

compute_strides :: (shape: []s64) -> []s64 {
  stride := NewArray(shape.count, s64);
  stride[shape.count-1] = 1;
  for < 0..shape.count-2 
    stride[it] = stride[it+1] * shape[it];
  return stride;
}

tensor :: (shape: []s64, $T: Type, allocator : Allocator = context.allocator) -> Tensor(T) {
  push_allocator(allocator);

  t: Tensor(T);
  t.shape     = array_copy(shape);
  t.stride    = compute_strides(shape);
  t.allocator = allocator;
  t.flags     = .CONTIGUOUS | .OWNED | .WRITEABLE;
  t.data      = NewArray(count(shape), T);
  return t;
}

tensor_like :: inline (a: Tensor($T), allocator : Allocator = context.allocator) -> Tensor(T) {
  return tensor_like(a, T, allocator);
}

tensor_like :: (a: Tensor($T), $R: Type, allocator : Allocator = context.allocator) -> Tensor(R) {
  push_allocator(allocator);

  t: Tensor(R);
  t.shape     = array_copy(a.shape);
  t.stride    = array_copy(a.stride);
  t.allocator = context.allocator;
  t.flags     = .CONTIGUOUS | .OWNED | .WRITEABLE;
  t.data      = NewArray(count(a.shape), T);
  return t;
}

tensor_free :: (t: *Tensor($T)) {
  if t.flags & .OWNED array_free(t.data);
  array_free(t.shape);
  array_free(t.stride);
}

shape_eq :: (a: Tensor($T), b: Tensor(T)) -> bool {
  if a.ndim != b.ndim return false;
  for a.shape if it != b.shape[it_index] return false;
  return true;
}

elements :: (el: $T, shape: []s64, allocator : Allocator = context.allocator) -> Tensor(T) {
  res := tensor(shape, T, allocator);
  for 0..res.count-1 res.data[it] = el;
  return res;
}

zeros :: (shape: []s64, $T: Type = f32, allocator : Allocator = context.allocator) -> Tensor(T) {
  return elements(cast(T) 0, shape, allocator);
}

ones :: (shape: []s64, $T: Type = f32, allocator : Allocator = context.allocator) -> Tensor(T) {
  return elements(cast(T) 1, shape, allocator);
}

range :: (shape: []s64, $T: Type = f32, allocator : Allocator = context.allocator) -> Tensor(T) {
  assert(cast(f64) max_of_type(T) >= cast(f64) (count(shape) - 1),
    "Range maximum value (%) exceeds % upper bound (%).", (count(shape) - 1), T , max_of_type(T));
  res := tensor(shape, T, allocator);
  for 0..res.count-1 res.data[it] = cast(T) it;
  return res;
}

random :: (shape: []s64, $T: Type = f32, allocator : Allocator = context.allocator) -> Tensor(T) {
  random :: #import "Random";
  assert(cast(f64) max_of_type(T) >= cast(f64) (count(shape) - 1),
    "Range maximum value (%) exceeds % upper bound (%).", (count(shape) - 1), T , max_of_type(T));
  res := tensor(shape, T, allocator);
  for 0..res.count-1 res.data[it] = cast(T) random.random_get();
  return res;
}

OP :: #string,\% JAI
operator %1 :: (a: Tensor($T), b: Tensor(T)) -> Tensor(%2) {
  assert(shape_eq(a, b), "Shape mismatched between lhs \% and rhs \%", a.shape, b.shape);
  c := tensor(a.shape, %2);
  for 0..a.count - 1 c.data[it] = a.data[it] %1 b.data[it];
  return c;
}

operator %1 :: (a: Tensor($T), b: T) -> Tensor(%2) {
  c := tensor(a.shape, %2);
  for 0..a.count - 1 c.data[it] = a.data[it] %1 b;
  return c;
}

operator %1 :: (b: T, a: Tensor($T)) -> Tensor(%2) {
  c := tensor(a.shape, %2);
  for 0..a.count - 1 c.data[it] = b %1 a.data[it];
  return c;
}
JAI

OP_MUT :: #string,\% JAI
operator %1= :: (a: *Tensor($T), b: Tensor(T)) {
  for 0..a.count - 1 a.data[it] %1= b.data[it];
}

operator %1= :: (a: *Tensor($T), b: T) {
  for 0..a.count - 1 a.data[it] %1= b.data[it];
}
JAI

FUN :: #string,\% JAI
%1 :: (a: Tensor($T), b: Tensor(T), allocator : Allocator = context.allocator) -> Tensor(T) {
  assert(shape_eq(a, b), "Shape mismatched between lhs \% and rhs \%", a.shape, b.shape);
  c := tensor_like(a, allocator = allocator);
  for 0..a.count - 1 c.data[it] = %1(a.data[it], b.data[it]);
  return c;
}

%1 :: (a: Tensor($T), b: T, allocator : Allocator = context.allocator) -> Tensor(T) {
  c := tensor_like(a, allocator = allocator);
  for 0..a.count - 1 c.data[it] = %1(a.data[it], b);
  return c;
}

%1 :: (b: T, a: Tensor($T), allocator : Allocator = context.allocator) -> Tensor(T) {
  c := tensor_like(a, allocator = allocator);
  for 0..a.count - 1 c.data[it] = %1(b, a.data[it]);
  return c;
}
JAI

UNARY :: #string,\% JAI
%1 :: (a: Tensor($T), allocator : Allocator = context.allocator) -> Tensor(T) {
  c := tensor(a.shape, T, allocator = allocator);
  for 0..a.count - 1 c.data[it] = %1(a.data[it]);
  return c;
}
JAI

using struct {
  #insert #run tprint(OP, "+", "T");
  #insert #run tprint(OP, "*", "T");
  #insert #run tprint(OP, "-", "T");
  #insert #run tprint(OP, "/", "T");
  #insert #run tprint(OP_MUT, "+", "T");
  #insert #run tprint(OP_MUT, "*", "T");
  #insert #run tprint(OP_MUT, "-", "T");
  #insert #run tprint(OP_MUT, "/", "T");
  #insert #run tprint(OP, "==", "bool");
  #insert #run tprint(OP, ">=", "bool");
  #insert #run tprint(OP, ">" , "bool");
  #insert #run tprint(OP, "<" , "bool");
  #insert #run tprint(OP, "<=", "bool");
  #insert #run tprint(FUN, "pow");
  #insert #run tprint(FUN, "atan2");
  #insert #run tprint(UNARY, "sin");
  #insert #run tprint(UNARY, "cos");
  #insert #run tprint(UNARY, "tan");
  #insert #run tprint(UNARY, "cot"); // this only works on f64 for some reason, even though tan does work on f32
  #insert #run tprint(UNARY, "atan");
  #insert #run tprint(UNARY, "abs");

  operator - :: (a: Tensor($T)) -> Tensor(T) {
    c := tensor(a.shape, T);
    for 0..a.count - 1 c.data[it] = -a.data[it];
    return c;
  }
};

as_type :: (a: Tensor($T), $R: Type) -> Tensor(R) {
  c := tensor(a.shape, R);
  for 0..a.count - 1 c.data[it] = cast(R) a.data[it];
  return c;
}

default_allocator :: Default_Allocator.allocator;

keep :: (a: *Tensor($T), allocator : Allocator = default_allocator) -> Tensor(T) {
  // if allocator == a.allocator return;
  t: Tensor(T) = tensor_like(a, allocator = allocator);
  memcpy(t.data.data, a.data.data, a.count);
  tensor_free(a);
  return t;
}

tensor_print :: (a: Tensor($T)) {
  push_allocator(default_allocator);
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


// TODO: i could just have this be index, this is just for semantics
Range :: struct {
  lo:   s64;
  hi:   s64; // exclusive
  step: s64;
}

END :s64: 0x7fff_ffff_ffff_ffff;

Index :: struct {
  using range: Range;
  #overlay (range.lo) idx: s64;
}

i :: (idx: s64) -> Index { return .{.{idx, 0, 0}}; }

i :: (lo: s64, hi: s64, step: s64 = 1) -> Index {
  assert(step != 0, "Range step cannot be 0");
  return .{.{lo, hi, step}};
}

wrap_index :: (using index: Index, dim_size: s64) -> Index {
  new_index := index;
  new_index.lo += xx (lo < 0) * dim_size;
  new_index.hi = ifx hi == END 
    then dim_size 
    else hi + xx (hi < 0) * dim_size;
    
  return new_index;
}

slice :: (a: Tensor($T), indices: ..Index) -> Tensor(T) {
  assert(indices.count <= a.ndim, "Too many indices (%) for tensor with ndim (%)", indices.count, a.ndim);

  is_contiguous := true;
  offset := 0;
  new_stride : [..] s64;
  new_shape  : [..] s64;

  for indices {
    dim_size := a.shape[it_index];
    using index := wrap_index(it, dim_size);

    if step == 0 {
      assert(index.idx >= 0 && idx < dim_size, 
        "Index % out of bounds for dim % (size %)", idx, it_index, dim_size);
      offset += index.idx * a.stride[it_index];
    } else {
      assert(step > 0 && lo >= 0 && hi <= dim_size && lo < hi ||
             step < 0 && lo < dim_size && hi >= 0  && lo > hi,
        "Invalid range [%, %) step % for dim % (size %)", lo, hi, step, it_index, dim_size);

      new_dim_size := (abs(hi - lo) + abs(step) - 1) / abs(step);
      offset += lo * a.stride[it_index];
      array_add(*new_shape, new_dim_size);
      array_add(*new_stride, a.stride[it_index] * step);
    }
  }

  for indices.count..a.ndim - 1 {
    array_add(*new_shape,  a.shape[it]);
    array_add(*new_stride, a.stride[it]);
  }

  t := a;
  t.data  += offset;
  t.shape  = new_shape;
  t.stride = new_stride;
  t.flags &= ~.OWNED;
  if !is_contiguous t.flags &= ~.CONTIGUOUS;

  return t;
}

benchmark :: () {
  N :: 1024;
  I :: 1000;
  allocator := context.allocator;
  push_allocator(temp);

  a := range(.[N, N], f64); 
  b := ones( .[N, N], f64, allocator); 
  c := ones( .[N, N], f64, allocator); 

  start := current_time_monotonic();
  for 1..I {
    d := a * b + c;
    a  = keep(*(d * b - c));
    reset_temporary_storage();
  }
  end := current_time_monotonic();
  elapsed := to_float64_seconds(end - start);
  total_elements := cast(f64) (N * N * I);
  print("Temporary_Storage uses % bytes.\n", context.temporary_storage.total_bytes_occupied);
  print("      Time: % s\n", elapsed);
  print("Throughput: % Mitems/s\n", total_elements / elapsed / 1_000_000);
}

main :: () {
  benchmark();
}

#import "Math";
#import "Basic";
Default_Allocator :: #import "Default_Allocator";
#import "Sloppy_Math";
#load   "util.jai";

// nice print function
//   -- maybe pad per column
// - vectorize
