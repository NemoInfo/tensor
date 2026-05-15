RawTensor :: struct (dtype: Type, shape: [] s64, stride: [] s64)
{
  data: [] dtype;
  #overlay (data.data)  ptr:   *dtype;
  #overlay (data.count) count: s64;
  ndim :: shape.count;
}

Tensor :: struct (dtype: Type, shape: [] s64, stride: [] s64, loc: string = "") { 
  using ref: *RawTensor(dtype, shape, stride);
}

compute_strides :: (shape: []s64) -> []s64 {
  if !shape.count return .[];
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

to_string :: (using loc: Source_Code_Location) -> string { return sprint("%:%:%", fully_pathed_filename, line_number, character_number); }

compute_tensor_type :: (E: Type, loc : string) -> Type { return compute_tensor_type(get_shape(E), get_dtype(E), loc); }
compute_tensor_type :: (shape: []s64, dtype: Type, loc: string) -> Type #compile_time {
  stride := compute_strides(shape);
  return #dynamic_specialize Tensor(dtype = dtype, shape = shape, stride = stride, loc = loc);
}

tensor_like :: (expr: $E, $loc := #caller_location, allocator := context.allocator) -> $RET #modify {
  RET = compute_tensor_type(E, to_string(loc));
  return true;
}{ return tensor(#run get_shape(E), #run get_dtype(E), loc, allocator); }

tensor :: ($shape: []s64, $dtype: Type = f32, $tid : string = "", allocator := context.allocator) -> $RET 
  #modify { RET = compute_tensor_type(shape, dtype, tid); return true; }
{
  res := New(RawTensor(RET.dtype, RET.shape, RET.stride),, allocator);
  res.data = NewArray(count(shape), dtype);
  return .{ res };
}

tensor :: ($shape: []s64, $dtype: Type = f32, $loc := #caller_location, allocator := context.allocator) -> $RET 
  #modify { RET = compute_tensor_type(shape, dtype, to_string(loc)); return true; }
{
  res := New(RawTensor(RET.dtype, RET.shape, RET.stride),, allocator);
  res.data = NewArray(count(shape), dtype);
  return .{ res };
}

zeros :: tensor;

element :: ($shape: []s64, el: $dtype, $loc := #caller_location, allocator := context.allocator) -> $RET 
  #modify { RET = compute_tensor_type(shape, dtype, to_string(loc)); return true; }
{
  res := tensor(shape, dtype, loc, allocator);
  for *res.data  it.* = el;
  return res;
}

ones :: ($shape: []s64, $dtype: Type = f32, $loc := #caller_location, allocator := context.allocator) -> $RET 
  #modify { RET = compute_tensor_type(shape, dtype, to_string(loc)); return true; }
{ return element(shape, cast(dtype) 1, loc); }

range :: ($shape: []s64, $dtype: Type = f32, $loc := #caller_location, allocator := context.allocator) -> $RET 
  #modify { RET = compute_tensor_type(shape, dtype, to_string(loc)); return true; } 
{
  res := tensor(shape, dtype, loc, allocator);
  res.data = NewArray(count(shape), dtype);
  for *res.data  it.* = cast(dtype) it_index;
  return res;
}

tensor_print :: (ts: Tensor($dtype, $shape, $stride, $loc)) { tensor_print(ts.ref); }
tensor_print :: (ts: RawTensor($dtype, $shape, $stride)) {
  if ts.ndim == 0  { print("Tensor(%)\n", ts.data[0]); return; }
  print_dim :: (data: [] dtype, dim: int, offset: s64, max_width: s64) {
    print("[");
    for 0..shape[dim]-1 {
      if dim == shape.count - 1 {
        if it > 0 print (" ");
        s := tprint("%", data[offset + it * stride[dim]]); 
        for 0..max_width-s.count-1 print(" ");
        print("%", s);
      } else {
        if it > 0 {
          print(",");
          for 0..shape.count-dim-2 print("\n");
          for 0..dim print(" ");
        }
        print_dim(data, dim + 1, offset + it * stride[dim], max_width);
      }
    }
    print("]");
  }

  max_width := 0;
  for 0..ts.count-1 {
    w := tprint("%", ts.data[it]);
    if w.count > max_width  max_width = w.count;
  }

  print_dim(ts.data, 0, 0, max_width);
  print("\n");
}

OpKind :: enum s32 {
  ADD;
  SUB;
  MUL;
  DIV;
  POW;
}

op_symbol :: (kind: OpKind) -> string {
  if kind == {
    case .ADD; return "+";
    case .SUB; return "-";
    case .MUL; return "*";
    case .DIV; return "/";
    case .POW; return "pow";
    case; panic("No symbol for %", kind); return "";
  }
}

op_accumulator :: (kind: OpKind) -> string {
  if kind == {
    case .ADD; return "xx 0";
    case .SUB; return "xx 0";
    case .MUL; return "xx 1";
    case .DIV; return "xx 1";
    case; panic("No accumulator for %", kind); return "";
  }
}

op_format :: (kind: OpKind, lhs: string, rhs: string) -> string {
  if kind == {
    case .ADD; #through; case .SUB; #through; case .MUL; #through; case .DIV; 
      return tprint("% % %", lhs, op_symbol(kind), rhs);
    case .POW;
      return tprint("%(%, %)", op_symbol(kind), lhs, rhs);
    case; panic("No symbol for %", kind); return "";
  }
}

OpReduce :: struct(kind: OpKind, 
                   dtype: Type, 
                   atype: Type, 
                   shape: [] s64,
                   axes:  [] s64
                   /*myabe add keep_dims flag*/) {
  arg: atype;
  ndim :: shape.count;
}

OpBroadcast :: struct(kind: OpKind, 
                      dtype: Type, 
                      ltype: Type, 
                      rtype: Type, 
                      shape: [] s64,
                      axes:  [][2] s64) {
  lhs: ltype;
  rhs: rtype;
  ndim :: shape.count;
}

is_struct_named :: (node_type: Type, name: string) -> bool {
  ti := cast(*Type_Info) node_type;
  if ti.type != .STRUCT return false;
  tis := cast(*Type_Info_Struct) ti;
  return tis.name == name;
}
is_tensor :: (node_type: Type) -> bool { return is_struct_named(node_type, "Tensor"); }
is_op_broadcast :: (node_type: Type) -> bool { return is_struct_named(node_type, "OpBroadcast"); }
is_op_reduce :: (node_type: Type) -> bool { return is_struct_named(node_type, "OpReduce"); }

get_struct_param :: (T: Type, $name: string, $type: Type) -> type {
  uninit : type = ---;
  ti := cast(*Type_Info) T;
  if ti.type != .STRUCT { panic("not a struct"); return uninit; }
  tis := cast(*Type_Info_Struct) ti;
  for tis.specified_parameters if it.name == name
    return (.*) cast(*type) (tis.constant_storage.data + it.offset_into_constant_storage);
  panic("field % not found", name); return uninit;
}

get_loc   :: (T: Type) -> string { return get_struct_param(T, "loc", string); }
get_ltype :: (T: Type) -> Type   { return get_struct_param(T, "ltype", Type); }
get_rtype :: (T: Type) -> Type   { return get_struct_param(T, "rtype", Type); }
get_atype :: (T: Type) -> Type   { return get_struct_param(T, "atype", Type); }
get_kind  :: (T: Type) -> OpKind { return get_struct_param(T, "kind", OpKind); }

get_dtype :: (node_type: Type) -> Type {
  ti := cast(*Type_Info) node_type;
  if ti.type != .STRUCT return node_type;
  tis := cast(*Type_Info_Struct) ti;
  if tis.name == {
    case "Tensor";   #through;
    case "OpReduce"; #through;
    case "OpBroadcast";
    for tis.specified_parameters {
      if it.name != "dtype" continue;
      return (cast(*Type) (tis.constant_storage.data + it.offset_into_constant_storage)).*;
    }
  }
  return node_type;
}

get_struct_array_param :: (T: Type, name: string, $type: Type, default: [] type = .[], throw := false) -> [] type {
  ti := cast(*Type_Info) T;
  if ti.type != .STRUCT {
    if throw  panic("Expected .STRUCT but found %", ti.type);
    return default;
  }
  tis := cast(*Type_Info_Struct) ti;
  for tis.specified_parameters {
    if it.name != name continue;
    tia := cast(*Type_Info_Array) it.type;
    res: [] type;
    if tia.array_type == .FIXED {
      res.count = tia.array_count;
      res.data  = cast(*type) (tis.constant_storage.data + it.offset_into_constant_storage);
    } else {
      res.count = (.*) cast(*s64)  (tis.constant_storage.data + it.offset_into_constant_storage);
      res.data  = (.*) cast(**type) (tis.constant_storage.data + it.offset_into_constant_storage + 8);
    }
    return res;
  }
  if throw  panic("Could not find parameter named % on type %", name, T);
  return default;
}

get_broadcast_axes :: (T: Type) -> [][2] s64 { return get_struct_array_param(T, "axes", [2]s64, throw = true); }
get_reduce_axes    :: (T: Type) -> [] s64    { return get_struct_array_param(T, "axes", s64,    throw = true); }
get_stride :: (T: Type) -> [] s64    { return get_struct_array_param(T, "stride", s64, .[1]); }
get_shape  :: (T: Type) -> [] s64    { return get_struct_array_param(T, "shape", s64, .[1]); }
get_shape  :: (val: $T) -> [] s64    { return get_shape(T); }

broadcast_shapes :: (l_shape: [] s64, r_shape: [] s64) -> (success: bool, out_shape: [] s64, axes: [][2] s64) {
  ndim_out := max(l_shape.count, r_shape.count);
  out_shape := NewArray(ndim_out, s64);
  axes      := NewArray(ndim_out, [2] s64);
  for i : 0..ndim_out - 1 {
    il := l_shape.count - ndim_out + i;
    ir := r_shape.count - ndim_out + i;
    l_sz := ifx il >= 0 then l_shape[il] else 1;
    r_sz := ifx ir >= 0 then r_shape[ir] else 1;
    if l_sz != 1 && r_sz != 1 && l_sz != r_sz  return false, .[], .[];
    out_sz := max(l_sz, r_sz);
    out_shape[i] = out_sz;
    axes[i][0] = ifx l_sz == out_sz && il >= 0 then il else -1;
    axes[i][1] = ifx r_sz == out_sz && ir >= 0 then ir else -1;
  }
  return true, out_shape, axes;
}

OP2 :: #string,\% JAI
%1 :: inline (l: $ltype, r: $rtype) -> $RET #modify {
  l_dtype := get_dtype(ltype);
  r_dtype := get_dtype(rtype);
  if l_dtype != r_dtype  return false, tprint("Could not %2 mismatched types (\%, \%)", l_dtype, r_dtype);

  l_shape := get_shape(ltype);
  r_shape := get_shape(rtype);
  success, shape, axes := broadcast_shapes(l_shape, r_shape);
  if !success  return false, tprint("Could not broadcast %2 mismatched shapes (\%, \%)", l_shape, r_shape);

  RET = #dynamic_specialize 
    OpBroadcast(kind = .%2, dtype = l_dtype, ltype = ltype, rtype = rtype, shape = shape, axes = axes);
  return true; 
} { return .{l, r}; }
JAI

sum :: (x: $T, $axes: [] s64) -> $RET 
  #modify {
    quick_sort(axes, x => x);
    shape := get_shape(T);
    new_shape := array_copy(shape);
    for < axes  array_ordered_remove_by_index(*new_shape, it);

    RET = #dynamic_specialize 
      OpReduce(kind = .ADD, dtype = get_dtype(T), atype = T, shape = new_shape, axes = axes);
    return true;
} { return .{x}; }

prod :: (x: $T, $axes: [] s64) -> $RET 
  #modify {
    quick_sort(axes, x => x);
    shape := get_shape(T);
    new_shape := array_copy(shape);
    for < axes  array_ordered_remove_by_index(*new_shape, it);

    RET = #dynamic_specialize 
      OpReduce(kind = .MUL, dtype = get_dtype(T), atype = T, shape = new_shape, axes = axes);
    return true;
} { return .{x}; }

using struct {
  #import "Math";
  #insert #run tprint(OP2, "operator +", "ADD");
  #insert #run tprint(OP2, "operator -", "SUB");
  #insert #run tprint(OP2, "operator *", "MUL");
  #insert #run tprint(OP2, "operator /", "DIV");
  #insert #run tprint(OP2, "pow", "POW");
}

_compute :: inline (expr: $E, $simd := false, $tid : string = "", allocator := context.allocator) 
  -> $RET #modify { RET = compute_tensor_type(E, tid); return true; } 
{ // @ASSUMES OpReduce is not child of expression
  ret := tensor(RET.shape, RET.dtype, tid, allocator);
  _compute(expr, *ret, simd);
  return ret;
}

_compute :: inline (expr: $E, $simd := false, $loc := #caller_location, allocator := context.allocator) 
  -> $RET #modify { RET = compute_tensor_type(E, to_string(loc)); return true; } 
{ // @ASSUMES OpReduce is not child of expression
  ret := tensor(RET.shape, RET.dtype, loc, allocator);
  _compute(expr, *ret, simd);
  return ret;
}

_compute :: (expr: $E, ret: *$RET, $simd := false)
  #modify { return RET == compute_tensor_type(E, get_loc(RET)); } 
{ // @ASSUMES OpReduce is not child of expression
  ret_ptr := ret.data;
  #if #run is_op_reduce(E) {
    code :: #run gen_reduce_code(E);
    #insert code;
  } else {
    code :: #run gen_broadcast_code(E);
    #insert code;
  }
}

compute :: inline (expr: $E, $simd := false, $loc := #caller_location, allocator := context.allocator) 
  -> $RET #modify { RET = compute_tensor_type(E, to_string(loc)); return true; } 
{
  ret := tensor(RET.shape, RET.dtype, loc, allocator);
  compute(expr, *ret, simd);
  return ret;
}

compute :: (expr: $E, ret: *$RET, $simd := false)
  #modify { return RET == compute_tensor_type(E, get_loc(RET)); } 
{ #insert #run -> string {
  sb: String_Builder;
  before, binding, type := gen_compute_fn_calls(E, root = true);

  print_to_builder(*sb, "%", before);
  print_to_builder(*sb, "_compute(%, ret);\n", binding);

  code := builder_to_string(*sb);
  println("================\n%\n================", code);
  return code;
};}

TensorId :: string;
Path :: struct {
  ENCODING :: string.[".lhs", ".rhs", ".arg"];

  lhs :: (path: [] u8) -> [] u8 {
    new_path := NewArray(path.count + 1, u8);
    memcpy(new_path.data, path.data, path.count * size_of(u8));
    new_path[path.count] = 0;
    return new_path;
  }
  
  rhs :: (path: [] u8) -> [] u8 {
    new_path := NewArray(path.count + 1, u8);
    memcpy(new_path.data, path.data, path.count * size_of(u8));
    new_path[path.count] = 1;
    return new_path;
  }

  arg :: (path: [] u8) -> [] u8 {
    new_path := NewArray(path.count + 1, u8);
    memcpy(new_path.data, path.data, path.count * size_of(u8));
    new_path[path.count] = 2;
    return new_path;
  }

  to_suffix :: (path: [] u8, end := ".ptr") -> string {
    sb: String_Builder;      
    for path  append(*sb, ENCODING[it]);
    append(*sb, end);
    return builder_to_string(*sb);
  }

  to_string :: (path: [] u8) -> string {
    sb: String_Builder;      
    for path  print_to_builder(*sb, "%", string.["l", "r", "a"][it]);
    return builder_to_string(*sb);
  }
}


Expression_Info :: struct {
  path_to_bs_tid: Table([]u8, Broadcast_TensorId, given_compare_function = array_eq);
  bs_to_idx:      Table([]s64, s64, given_compare_function = array_eq);
  tid_to_suf_tb: Table(TensorId, Suffix_TensorBinding);

  Broadcast_TensorId   :: struct { bs: []s64; tid: TensorId; }
  Index_TensorId       :: struct { idx: s64; tid: TensorId; }
  Suffix_TensorBinding :: struct { suf: string; tb: s64; }
  Path_to_Bs_Tid       :: Table([]u8, Broadcast_TensorId, given_compare_function = array_eq);
}

expression_info_free :: (using e_info: *Expression_Info) {
  deinit(*path_to_bs_tid);
  deinit(*bs_to_idx);
  deinit(*tid_to_suf_tb);
}

path_to_tb_idx :: (e_info: *Expression_Info, path: [] u8) -> (tb: s64, idx: s64) {
  f0, bs_tid    := table_find(*e_info.path_to_bs_tid, path); assert(f0);
  f1, idx       := table_find(*e_info.bs_to_idx, bs_tid.bs); assert(f1);
  f2, suffix_tb := table_find(*e_info.tid_to_suf_tb, bs_tid.tid); assert(f2);
  return suffix_tb.tb, idx;
}

collect_expression_info :: (E: Type) -> Expression_Info {
  e_info: Expression_Info;
  collect_path_to_bs_tid(E, *e_info.path_to_bs_tid);

  for bs_tid, path: e_info.path_to_bs_tid {
    if !table_contains(*e_info.bs_to_idx, bs_tid.bs)
      table_add(*e_info.bs_to_idx, bs_tid.bs, e_info.bs_to_idx.count);
    _, idx := table_find(*e_info.bs_to_idx, bs_tid.bs);
    if !table_contains(*e_info.tid_to_suf_tb, bs_tid.tid)
      table_add(*e_info.tid_to_suf_tb, bs_tid.tid, 
                .{Path.to_suffix(path), e_info.tid_to_suf_tb.count});
  }

  return e_info;
}

collect_path_to_bs_tid :: (E: Type, path_to_bs_tid: *Expression_Info.Path_to_Bs_Tid , path: []u8 = .[]) {
  ti := cast(*Type_Info) E;
  if ti.type != .STRUCT return;
  tis := cast(*Type_Info_Struct) ti;
  if tis.name == {
    case "Tensor";
      stride := get_stride(E);
      tid    := get_loc(E);
      table_add(path_to_bs_tid, path, .{stride, tid});
    case "OpReduce"; using Path;
      atype := get_atype(E);
      collect_path_to_bs_tid(atype, path_to_bs_tid, arg(path));
    case "OpBroadcast"; using Path;
      ltype := get_ltype(E);
      rtype := get_rtype(E);
      shape := get_shape(E);
      ndim := shape.count;
      axes := get_broadcast_axes(E);

      l_table: Expression_Info.Path_to_Bs_Tid; defer deinit(*l_table);
      r_table: Expression_Info.Path_to_Bs_Tid; defer deinit(*r_table);
      collect_path_to_bs_tid(ltype, *l_table, lhs(path));
      collect_path_to_bs_tid(rtype, *r_table, rhs(path));

      transform :: (child_stride: []s64, side: s64, axes: [][2]s64, ndim: s64) -> []s64 {
        bs := NewArray(ndim, s64);
        for dim: 0..ndim-1 {
          ax := axes[dim][side];
          bs[dim] = ifx ax >= 0 then child_stride[ax] else 0;
        }
        return bs;
      }

      for *bs_tid, child_path: l_table
        table_add(path_to_bs_tid, child_path, .{transform(bs_tid.bs, 0, axes, ndim), bs_tid.tid});

      for *bs_tid, child_path: r_table
        table_add(path_to_bs_tid, child_path, .{transform(bs_tid.bs, 1, axes, ndim), bs_tid.tid});
  }
}

generate_compute_string :: (E: Type, e_info: *Expression_Info, path: []u8 = .[], $root := false) -> (before: string, binding: string) {
  ti := cast(*Type_Info) E;
  if ti.type != .STRUCT { todo("not implemented"); return "", ""; }
  tis := cast(*Type_Info_Struct) ti;
  if tis.name == {
    case "Tensor";
      tb, idx := path_to_tb_idx(e_info, path);
      return "", tprint("t%[idx%]", tb, idx);
    case "OpReduce";
      a_before, arg := generate_compute_string(get_atype(E), e_info, Path.arg(path), root);
      return a_before, arg;
    case "OpBroadcast";
      kind := get_kind(E);
      l_before, lhs := generate_compute_string(get_ltype(E), e_info, Path.lhs(path));
      r_before, rhs := generate_compute_string(get_rtype(E), e_info, Path.rhs(path));
      operation := op_format(kind, lhs, rhs); 

      if root return tprint("%%", l_before, r_before), operation;

      binding := tprint("inter_%", Path.to_string(path));
      before  := tprint("%%% := %;\n", l_before, r_before, binding, operation);
      return before, binding;
  }
  todo("not implemented"); return "", "";
}

// @PERFORMANCE: seems to be faster than a Table since we don't expect to have many 
// of these active at a time, also the *void on the type might be a bit specific since 
// it will probably only reuse tensors that have literally the exact same role
TENSOR_CACHE : [..] struct {key: s64; ptr: *void;};
tensor_cache_free :: () {
  for TENSOR_CACHE free(it.ptr);
  array_free(TENSOR_CACHE);
}
tensor_cache_find :: inline (key: s64) -> bool, *void {
  for TENSOR_CACHE if it.key == key return true, it.ptr;
  return false, null;
}
gen_compute_fn_calls :: (E: Type, path: []u8 = .[], $root := false) -> (before: string, binding: string, type: Type) #compile_time {
  ti := cast(*Type_Info) E;
  if ti.type != .STRUCT { todo("not implemented"); return "", "", void; }
  tis := cast(*Type_Info_Struct) ti;
  if tis.name == {
    case "Tensor";
      return "", tprint("expr%", Path.to_suffix(path, "")), E;
    case "OpReduce"; // @TODO: go shopping for spaces to put tensors in greedy free
      a_before, arg, atype := gen_compute_fn_calls(get_atype(E), Path.arg(path));
      type := #dynamic_specialize OpReduce(kind=get_kind(E), dtype=get_dtype(E), atype=atype, shape=get_shape(E), axes=get_reduce_axes(E));
      type_str := insertable_type_string(type);
      operation := tprint("%.{%}", type_str, arg); 
      binding := tprint("%", Path.to_string(path));
      if root return a_before, operation, void;
      otype := compute_tensor_type(get_shape(E), get_dtype(E), binding);
      before  := tprint(#string JAI
%1%2_expr := %3;
found_%2, %2_ptr := tensor_cache_find(0x%4);
%2 := ifx !found_%2 _compute(%2_expr, tid = "%2")
      else          .{%2_ptr}; 
if found_%2 _compute(%2_expr, *%2);
else        array_add(*TENSOR_CACHE, .{0x%4, cast(*void) %2.ref});
JAI, a_before, binding, operation, *otype);
      return before, binding, otype;
    case "OpBroadcast";
      kind := get_kind(E);
      l_before, lhs, ltype := gen_compute_fn_calls(get_ltype(E), Path.lhs(path));
      r_before, rhs, rtype := gen_compute_fn_calls(get_rtype(E), Path.rhs(path));
      type := #dynamic_specialize OpBroadcast(kind=get_kind(E), dtype=get_dtype(E), ltype=ltype, rtype=rtype, shape=get_shape(E), axes=get_broadcast_axes(E));
      type_str := insertable_type_string(type);
      operation := tprint("%.{%, %}", type_str, lhs, rhs); 

      if root return tprint("%%", l_before, r_before), operation, void;

      binding := tprint("%_expr", Path.to_string(path));
      before  := tprint("%%% := %;\n", l_before, r_before, binding, operation);
      return before, binding, type;
  }
  todo("not implemented"); return "", "", void;
}

// @ASSUMES: output pointer exists in ret_ptr
// @ASSUMES: no nested OpReduce operations
gen_reduce_code :: (E: Type) -> (code: string) #compile_time {

  sb: String_Builder;
  dtype  := get_dtype(E);
  atype  := get_atype(E);
  shape  := get_shape(E);
  axes   := get_reduce_axes(E);
  kind   := get_kind(E);
  ashape := get_shape(atype);
  andim  := ashape.count;
  e_info := collect_expression_info(E); defer expression_info_free(*e_info);
  // for e_info.path_to_bs_tid println("%: %", it_index, it); println(""); for e_info.bs_to_idx println("%: %", it_index, it); println(""); for e_info.tid_to_suf_tb println("%: %", it_index, it); println("");
  
  for suf_tb, idx_tid: e_info.tid_to_suf_tb
    print_to_builder(*sb, "t% := expr%;\n", suf_tb.tb, suf_tb.suf);

  print_to_builder(*sb, "out_idx := 0;\n");

  is_reduced := NewArray(andim, bool); defer array_free(is_reduced);
  for axes  is_reduced[it] = true;

  if is_reduced[andim-1] /* || true */ {
    for sh, dim: ashape if !is_reduced[dim]
      print_to_builder(*sb, "for i%: 0..% {\n", dim, sh - 1);

    print_to_builder(*sb, "acc : % = %;\n", dtype, op_accumulator(kind));
  
    for sh, dim: ashape if is_reduced[dim]
      print_to_builder(*sb, "for i%: 0..% {\n", dim, sh - 1);

    // @TODO use join here so i don't have to print the 0
    for idx, bs : e_info.bs_to_idx {
      print_to_builder(*sb, "idx% := 0", idx);
      for dim : 0 .. andim-1
        if bs[dim] != 0
          print_to_builder(*sb, " + i% * %", dim, bs[dim]);
      append(*sb, ";\n");
    }

    before, binding := generate_compute_string(E, *e_info, root = true);
    print_to_builder(*sb, "%", before);
    print_to_builder(*sb, "acc %= %;\n", op_symbol(kind), binding);

    for sh, dim: ashape if is_reduced[dim] append(*sb, "}\n");

    append(*sb, "ret_ptr[out_idx] = acc;\n");
    append(*sb, "out_idx += 1;\n");

    for sh, dim: ashape if !is_reduced[dim] append(*sb, "}\n");
  } else {
    stride := compute_strides(shape);
    out_idx_parts := NewArray(shape.count, string); defer(*out_idx_parts);
    odim := 0;
    print_to_builder(*sb, "for 0..%  ret_ptr[it] = %;\n", count(shape) - 1, op_accumulator(kind));
    for adim: 0..andim-1
      if !is_reduced[adim] {
        if stride[odim] == 1
          out_idx_parts[odim] = tprint("i%", adim);
        else
          out_idx_parts[odim] = tprint("i% * %", adim, stride[odim]);
        odim += 1;
      }
    out_idx_expr := String.join(..out_idx_parts, " + ");

    for sh, dim: ashape  print_to_builder(*sb, "for i%: 0..% {\n", dim, sh - 1);
    for idx, bs : e_info.bs_to_idx {
      print_to_builder(*sb, "idx% := 0", idx);
      for dim : 0 .. andim-1
        if bs[dim] != 0
          print_to_builder(*sb, " + i% * %", dim, bs[dim]);
      append(*sb, ";\n");
    }

    before, binding := generate_compute_string(E, *e_info, root = true);
    print_to_builder(*sb, "%", before);
    print_to_builder(*sb, "ret_ptr[%] %= %;\n", out_idx_expr, op_symbol(kind), binding);


    for sh, dim: ashape  append(*sb, "}\n");
  }

  code := builder_to_string(*sb);
  println("%", code);
  return code;
}

gen_broadcast_code :: (E: Type) -> (code: string) #compile_time {
  // @ASSUMES: output pointer exists in ret_ptr
  // @ASSUMES: no nested OpReduce operations
  // println("%\n", E);

  e_info := collect_expression_info(E); defer expression_info_free(*e_info);
  // for e_info.path_to_bs_tid println("%: %", it_index, it); println(""); for e_info.bs_to_idx println("%: %", it_index, it); println(""); for e_info.tid_to_suf_tb println("%: %", it_index, it); println("");
  shape := get_shape(E);
  dtype := get_dtype(E);
  ndim  := shape.count;
  sb: String_Builder;
  for suf_tb, idx_tid: e_info.tid_to_suf_tb
    print_to_builder(*sb, "t% : *% = expr%;\n", suf_tb.tb, dtype, suf_tb.suf);

  print_to_builder(*sb, "out_idx := 0;\n");

  // @TODO: maybe keep a single loop and do modulus on out_idx / shape instead
  for sh, dim: shape
  if sh > 1  print_to_builder(*sb, "for i%: 0..% {\n", dim, sh - 1);
  else       print_to_builder(*sb, "i% := 0;\n", sh);

  for idx, bs: e_info.bs_to_idx {
    add_parts: [..] string; defer array_free(add_parts);
    for dim : 0..ndim-1 if bs[dim] != 0  
      if bs[dim] == 1  array_add(*add_parts, tprint("i%", dim));
      else array_add(*add_parts, tprint("i% * %", dim, bs[dim]));
    print_to_builder(*sb, "idx% := %;\n", idx, String.join(..add_parts, " + "));
  }

  before, binding := generate_compute_string(E, *e_info, root = true);
  print_to_builder(*sb, "%ret_ptr[out_idx] = %;\n", before, binding);

  print_to_builder(*sb, "out_idx += 1;\n");
  for dim: 0..ndim - 1 if shape[dim] > 1  print_to_builder(*sb, "}\n");

  code := builder_to_string(*sb);
  println("%", code);
  return code;
}

main :: () {
  N :: 1024;
  a := range(.[N, N], f32); defer free(a.ref);
  b := ones( .[N], f32); defer free(b.ref);
  c := ones( .[N, N], f32); defer free(c.ref);
  expr0 := sum(a, .[1]);
  // expr1 := (a*sum(c + b, .[0]) + a*b) * (a/b + b*c);
  // expr2 := sum(expr1, axes = .[0, 1]);
  ret0  := tensor_like(expr0); defer free(ret0.ref);
  // ret1  := tensor_like(expr1); defer free(ret1.ref);
  // ret2  := tensor_like(expr2); defer free(ret1.ref);
  benchmark_function(compute(expr0, *ret0));
  // benchmark_function(compute(expr1, *ret1));
  // benchmark_function(compute(expr2, *ret2));
}

#import "Basic";
#import "Hash_Table";
String :: #import "String";
#import "Sort";
Hash :: #import "Hash";
#load "util.jai";

benchmark_function :: (fn: Code, iterations := 1000) #expand {
  fn_name :: #run center_pad(print_code(fn), 80, #char "-");
  start := current_time_monotonic();
  for 1..iterations #insert fn;
  dt := to_float64_seconds(current_time_monotonic() - start);
  res := tprint("% it/s (%ms elapsed)", iterations / dt, formatInt(cast (s64) (dt * 1000), digits_per_comma=3, comma_string = ","));
  println("+%+\n|%|\n+%+", fn_name, center_pad(res, 80), center_pad("", 80, #char "-"));
}
