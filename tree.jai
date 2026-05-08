OpKind :: enum {
  ADD;
  SUB;
  MUL;
  DIV;

  POW;
  SIN;
  COS;
  TAN;
  COT;
}

Op1 :: struct($K: OpKind, $T: Type, $C: Type) {
  arg: C;
}

Op2 :: struct($K: OpKind, $T: Type, $L: Type, $R: Type, $Cost: s64) {
  lhs: L;
  rhs: R;
}

operator - :: (arg: $C) -> Op1(.SUB, $T, C) #modify {
  T = get_dtype(C);
  return true; 
} { return .{arg}; } 

operator + :: (arg: $C) -> C { return arg; } 

OP1 :: #string,\% JAI
%1 :: (c: $C) -> Op1(.%2, $T, C) #modify { 
  T = get_dtype(C); 
  return T != C; 
} { return .{c}; }
JAI

OP2 :: #string,\% JAI
%1 :: (l: $L, r: $R) -> $RET #modify {
  if !is_node(L) && !is_node(R) return false;

  l_dtype := get_dtype(L);
  r_dtype := get_dtype(R);
  if l_dtype != r_dtype return false, tprint("lhs dtype != rhs dtype mismatch \% != \%", l_dtype, r_dtype);

  l_cost := get_cost(L);
  r_cost := get_cost(R);

  cost := ifx l_cost == r_cost then l_cost + 1 else max(l_cost, r_cost);

  RET = #dynamic_specialize Op2(K = .%2, T = l_dtype, L = L, R = R, Cost = cost);
  return true; 
} { return .{l, r}; }
JAI

// @TODO: polish this
OP2EQ :: #string,\% JAI
operator %1= :: (x: *Tensor($T), y: $O) { compute_simd(x.* %1 y, x); }
JAI

using struct {
  #import "Math";

  #insert #run tprint(OP2, "operator +", "ADD");
  #insert #run tprint(OP2, "operator -", "SUB");
  #insert #run tprint(OP2, "operator *", "MUL");
  #insert #run tprint(OP2, "operator /", "DIV");
  #insert #run tprint(OP2, "pow", "POW");

  #insert #run tprint(OP1, "sin", "SIN");
  #insert #run tprint(OP1, "cos", "COS");
  #insert #run tprint(OP1, "tan", "TAN");
  #insert #run tprint(OP1, "cot", "COT");

  #insert #run tprint(OP2EQ, "+");
  #insert #run tprint(OP2EQ, "-");
  #insert #run tprint(OP2EQ, "*");
  #insert #run tprint(OP2EQ, "/");
}

collect_tensors :: (E: Type) -> [] string #compile_time {
  tensors : [..] string;
  collect_tensors(E, *tensors);
  return tensors;
}

collect_tensors :: (E: Type, tensors: *[..] string, path := "") #compile_time {
  ti := cast(*Type_Info) E;
  if ti.type != .STRUCT return;
  tis := cast(*Type_Info_Struct) ti;
  if tis.name == {
    case "Tensor"; array_add(tensors, sprint("%.ptr", path));
    case "Op1"; for tis.specified_parameters if it.name == "C" {
      arg_type := (cast(*Type) (tis.constant_storage.data + it.offset_into_constant_storage)).*;
      collect_tensors(arg_type, tensors, sprint("%.arg", path));
    }
    case "Op2"; 
      for tis.specified_parameters if it.name == {
        case "L";
          arg_type := (cast(*Type) (tis.constant_storage.data + it.offset_into_constant_storage)).*;
          collect_tensors(arg_type, tensors, sprint("%.lhs", path));
        case "R";
          arg_type := (cast(*Type) (tis.constant_storage.data + it.offset_into_constant_storage)).*;
          collect_tensors(arg_type, tensors, sprint("%.rhs", path));
      }
  }
}

collect_pointer_declarations :: (E: Type) -> string #compile_time {
  sb: String_Builder;
  for collect_tensors(E) print_to_builder(*sb, "t% := expr%;\n", it_index, it);
  return builder_to_string(*sb);
}

is_node :: (S: Type) -> bool #compile_time {
  ti := cast(*Type_Info) S;
  if ti.type != .STRUCT return false;
  tis := cast(*Type_Info_Struct) ti;
  if tis.name == {
    case "Tensor"; #through;
    case "Op1";    #through;
    case "Op2"; return true;
  }
  return false;
}

get_dtype :: (S: Type) -> Type #compile_time {
  ti := cast(*Type_Info) S;
  if ti.type != .STRUCT return S;
  tis := cast(*Type_Info_Struct) ti;
  if tis.name == {
    case "Tensor"; #through;
    case "Op1";  #through;
    case "Op2";
    for tis.specified_parameters {
      if it.name != "T" continue;
      return (cast(*Type) (tis.constant_storage.data + it.offset_into_constant_storage)).*;
    }
  }
  return S;
}

get_cost :: (S: Type) -> s64 #compile_time {
  ti := cast(*Type_Info) S;
  if ti.type != .STRUCT return 1;
  tis := cast(*Type_Info_Struct) ti;
  if tis.name == {
    case "Tensor"; return 0;
    case "Op1"; panic("unreachable");
    case "Op2"; 
    for tis.specified_parameters {
      if it.name != "Cost" continue;
      return (cast(*s64) (tis.constant_storage.data + it.offset_into_constant_storage)).*;
    }
  }
  return -1;
}

eval_it :: inline (val: $T, nd_index: []s64) -> T #modify { return T == get_dtype(T); } { return val; }

eval_it :: inline (tensor: Tensor($T), nd_index: []s64) -> T {
  flat := 0;
  offset := nd_index.count - tensor.ndim;
  for 0..tensor.ndim-1 {
    idx := nd_index[it + offset];
    if tensor.shape[it] == 1 then idx = 0;
    flat += idx * tensor.stride[it];
  }
  return tensor.data[flat];
}

eval_it :: inline (expr: Op1($K, $T, $C), nd_index: []s64) -> T {
  c := eval_it(expr.arg, nd_index);
  #if K == {
    case .SUB; return -c;
    case .SIN; return sin(c);
    case .COS; return cos(c);
    case .TAN; return tan(c);
    case .COT; return cot(c);
  }
}

eval_it :: inline (expr: Op2($K, $T, $L, $R, $Cost), nd_index: []s64) -> T {
  l := eval_it(expr.lhs, nd_index);
  r := eval_it(expr.rhs, nd_index);
  #if K == {
    case .ADD; return l + r;
    case .SUB; return l - r;
    case .MUL; return l * r;
    case .DIV; return l / r;
    case .POW; return pow(l, r);
  }
}

broadcast_shape :: (a: []s64, b: []s64) -> []s64 {
  ndim := max(a.count, b.count);
  result := NewArray(ndim, s64);
  for 0..ndim-1 {
    da := ifx (a.count - ndim + it) >= 0 then a[a.count - ndim + it] else 1;
    db := ifx (b.count - ndim + it) >= 0 then b[b.count - ndim + it] else 1;
    assert(da == db || da == 1 || db == 1, "Shapes not broadcastable: % vs %", a, b);
    result[it] = max(da, db);
  }
  return result;
}

shape_of :: (expr: Op2($K, $T, $L, $R, $Cost)) -> []s64 {
  return broadcast_shape(shape_of(expr.lhs), shape_of(expr.rhs));
}
shape_of :: (expr: Op1($K, $T, $C))     -> []s64 { return shape_of(expr.arg); }
shape_of :: (tensor: Tensor($T))      -> []s64 { return tensor.shape; }
shape_of :: (expr: $T) -> []s64 #modify { return T == get_dtype(T); } { return .[]; }

is_broadcasting :: (expr: $E, output_shape: []s64) -> bool #modify { return E == get_dtype(E); } {
  return !shape_eq(shape_of(expr), output_shape);
}

is_broadcasting :: (t: Tensor($T), output_shape: []s64) -> bool {
  return !shape_eq(t.shape, output_shape);
}

is_broadcasting :: (expr: Op1($K, $T, $C), output_shape: []s64) -> bool {
  return is_broadcasting(expr.arg, output_shape);
}

is_broadcasting :: (expr: Op2($K, $T, $L, $R, $Cost), output_shape: []s64) -> bool {
  return is_broadcasting(expr.lhs, output_shape) || 
         is_broadcasting(expr.rhs, output_shape);
}

compute :: inline (expr: $E) -> Tensor($T) #modify {
  T = get_dtype(E);
  return true;
} {
  shape := shape_of(expr);
  ret   := tensor(shape, T);

  compute(expr, *ret);

  return ret;
}

compute :: inline (expr: $E, ret: *Tensor($T)) #modify {
  T = get_dtype(E);
  return true;
} {
  shape := shape_of(expr);
  assert(shape_eq(shape, ret.shape));

  if is_broadcasting(expr, shape) {
    nd_index := NewArray(ret.ndim, s64,, temp);
    for 0..ret.count - 1 {
      tmp := it;
      for < 0..ret.ndim - 1 {
        nd_index[it] = tmp % ret.shape[it];
        tmp /= ret.shape[it];
      }
      ret.data[it] = eval_it(expr, nd_index);
    }
  } else {
    for 0..ret.count - 1 ret.data[it] = eval_it(expr, it);
  }
}

compute_simd :: inline (expr: $E) -> Tensor($T) #modify {
  T = get_dtype(E);
  return true;
} {
  shape := shape_of(expr);
  ret   := tensor(shape, T);

  compute_simd(expr, *ret);

  return ret;
}

compute_simd :: inline (expr: $E, ret: *Tensor($T)) #modify {
  T = get_dtype(E);
  return true;
} {
  #import "File";

  MAX_COST  :: #run get_cost(E);
  REG_COUNT :: #run min(MAX_COST + 1, 16);

  simd_n := ret.count & ~(cast(s64) 7);
  ret_ptr := ret.ptr;
  i := 0;

  tensors :: #run collect_tensors(E);

  // @TODO: I should verify if this pointer math is really slowing me down
  //        we might be able to actually get away with it
  //        Consider having an array of pointers to index with a single
  //        layer of indirection, might be a good compromise ...

  #insert #run collect_pointer_declarations(E);
  #insert #run read_entire_file("simd.jai");

  #asm AVX {
    ymm0:  vec === 0;  ymm1:  vec === 1;  ymm2:  vec === 2;  ymm3:  vec === 3;
    ymm4:  vec === 4;  ymm5:  vec === 5;  ymm6:  vec === 6;  ymm7:  vec === 7;
    ymm8:  vec === 8;  ymm9:  vec === 9;  ymm10: vec === 10; ymm11: vec === 11;
    ymm12: vec === 12; ymm13: vec === 13; ymm14: vec === 14; ymm15: vec === 15;
  }
  #if REG_COUNT > 16 {
    panic("spill not implemented");
  }
  for 0..simd_n / 8 - 1 {
    eval_it_simd(*expr, 0);
    #asm AVX { movups.y [ret_ptr + i], ymm0; }
    i += size_of(T) * 8;
  } 

  for simd_n..ret.count - 1 ret.data[it] = eval_it(expr, it);
}

// @TODO: Consider turining these to macro expansions as well? might speed us
//        up since i dont have to deal with the stack pointer...

eval_it :: inline (t: Tensor($T), i: s64) -> T { 
  return t.data[i]; 
}
eval_it :: inline (val: $T, i: s64) -> T #modify { 
  return T == get_dtype(T); 
} { return val; }
eval_it :: inline (expr: Op1($K, $T, $C), i: s64) -> T {
  c := eval_it(expr.arg, i);
  #if K == {
    case .SUB; return -c;
    case .SIN; return sin(c);
    case .COS; return cos(c);
    case .TAN; return tan(c);
    case .COT; return cot(c);
  }
}
eval_it :: inline (expr: Op2($K, $T, $L, $R, $Cost), i: s64) -> T {
  l := eval_it(expr.lhs, i);
  r := eval_it(expr.rhs, i);
  #if K == {
    case .ADD; return l + r;
    case .SUB; return l - r;
    case .MUL; return l * r;
    case .DIV; return l / r;
    case .POW; return pow(l, r);
  }
}
