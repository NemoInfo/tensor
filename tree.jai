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
operator %1= :: (x: *Tensor($T, $Loc), y: $O) { compute_simd(x.* %1 y, x); }
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

// @TODO: also maybe i should just use two u128s or something, one for length
//        another as a binary op, that would give a tree that is 128 deep which
//        is crazy but still, have to think about it.
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

TensorPath :: struct {
  suffix: string;
  index:  [] u8;
}

TensorInfo :: struct {
  suffix: string;
  t_id: s64;
}

#import "Hash_Table";
to_string :: (using loc: Source_Code_Location) -> string { return sprint("%:%:%", fully_pathed_filename, line_number, character_number); }
TensorTable :: Table(string, TensorInfo);

collect_tensors :: (E: Type, tensors: *TensorTable, suffix := "") #compile_time {
  ti := cast(*Type_Info) E;
  if ti.type != .STRUCT return;
  tis := cast(*Type_Info_Struct) ti;
  if tis.name == {
    case "Tensor"; 
      for tis.specified_parameters if it.name == "Loc" {
        loc := (cast(*string) (tis.constant_storage.data + it.offset_into_constant_storage)).*;
        if !table_contains(tensors, loc)
          table_add(tensors, loc, .{ptr(suffix), 0});
        return;
      }

    case "Op1"; for tis.specified_parameters if it.name == "C" {
      arg_type := (cast(*Type) (tis.constant_storage.data + it.offset_into_constant_storage)).*;
      collect_tensors(arg_type, tensors, arg(suffix));
    }
    case "Op2"; 
      for tis.specified_parameters if it.name == {
        case "L";
          arg_type := (cast(*Type) (tis.constant_storage.data + it.offset_into_constant_storage)).*;
          collect_tensors(arg_type, tensors, lhs(suffix));
        case "R";
          arg_type := (cast(*Type) (tis.constant_storage.data + it.offset_into_constant_storage)).*;
          collect_tensors(arg_type, tensors, rhs(suffix));
      }
  }
}

collect_tensors :: (E: Type) -> TensorTable #compile_time {
  tensors : TensorTable;
  collect_tensors(E, *tensors);
  return tensors;
}

lhs :: (suffix: string) -> string { return sprint("%.lhs", suffix); }
rhs :: (suffix: string) -> string { return sprint("%.rhs", suffix); }
arg :: (suffix: string) -> string { return sprint("%.arg", suffix); }
ptr :: (suffix: string) -> string { return sprint("%.ptr", suffix); }

MAX_TENSOR_REGISTERS :: 16
                       - 1 // ret_ptr
                       - 1 // i               
                       - 1 // (stack pointer)
                       - 1 // (frame pointer)
                       - 1 // ptr
                       - 1 // broadcast
                       ;

collect_pointer_declarations :: (E: Type) -> string, TensorTable #compile_time {
  sb: String_Builder;
  tensors := collect_tensors(E);
  using_tensor_array := tensors.count > MAX_TENSOR_REGISTERS;
  tensor_registers := ifx using_tensor_array then MAX_TENSOR_REGISTERS - 1 else tensors.count;
  i := 0;
  for *val, key: tensors {
    if i < tensor_registers {
      print_to_builder(*sb, "t% := expr%;\n", i, val.suffix);
      val.t_id = i;
    } else { break; }
    i += 1;
  }
  if !using_tensor_array return builder_to_string(*sb), tensors;

  print_to_builder(*sb, "ts: [%] *% = .[", tensors.count - (MAX_TENSOR_REGISTERS - 1), get_dtype(E));
  i = 0;
  for val, key: *tensors {
    if i < tensor_registers { i += 1; continue; }
    print_to_builder(*sb, "expr%,\n", val.suffix);
    val.t_id = i;
    i += 1;
  }
  print_to_builder(*sb, "];\nts_ptr := ts.data;");

  return builder_to_string(*sb), tensors;
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

// lhs :: (S: Type) -> Type #compile_time {
//   ti := cast(*Type_Info) S;
//   assert(ti.type == .STRUCT);
//   tis := cast(*Type_Info_Struct) ti;
//   assert(tis.name == "Op2");
//   for tis.specified_parameters {
//     if it.name != "L" continue;
//     return (cast(*Type) (tis.constant_storage.data + it.offset_into_constant_storage)).*;
//   }
//   panic("unreachable"); 
// }
// 
// rhs :: (S: Type) -> Type #compile_time {
//   ti := cast(*Type_Info) S;
//   assert(ti.type == .STRUCT);
//   tis := cast(*Type_Info_Struct) ti;
//   assert(tis.name == "Op2");
//   for tis.specified_parameters {
//     if it.name != "R" continue;
//     return (cast(*Type) (tis.constant_storage.data + it.offset_into_constant_storage)).*;
//   }
//   panic("unreachable");
// }

loc :: (S: Type) -> string #compile_time {
  ti := cast(*Type_Info) S;
  assert(ti.type == .STRUCT);
  tis := cast(*Type_Info_Struct) ti;
  assert(tis.name == "Tensor");
  for tis.specified_parameters {
    if it.name != "Loc" continue;
    return (cast(*string) (tis.constant_storage.data + it.offset_into_constant_storage)).*;
  }
  panic("unreachable"); return "";
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

eval_it :: inline (tensor: Tensor($T, $Loc), nd_index: []s64) -> T {
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
shape_of :: (tensor: Tensor($T, $Loc))      -> []s64 { return tensor.shape; }
shape_of :: (expr: $T) -> []s64 #modify { return T == get_dtype(T); } { return .[]; }

is_broadcasting :: (expr: $E, output_shape: []s64) -> bool #modify { return E == get_dtype(E); } {
  return !shape_eq(shape_of(expr), output_shape);
}

is_broadcasting :: (t: Tensor($T, $Loc), output_shape: []s64) -> bool {
  return !shape_eq(t.shape, output_shape);
}

is_broadcasting :: (expr: Op1($K, $T, $C), output_shape: []s64) -> bool {
  return is_broadcasting(expr.arg, output_shape);
}

is_broadcasting :: (expr: Op2($K, $T, $L, $R, $Cost), output_shape: []s64) -> bool {
  return is_broadcasting(expr.lhs, output_shape) || 
         is_broadcasting(expr.rhs, output_shape);
}

compute :: inline (expr: $E) -> Tensor($T, $Loc) #modify {
  T = get_dtype(E);
  return true;
} {
  shape := shape_of(expr);
  ret   := tensor(shape, T);

  compute(expr, *ret);

  return ret;
}

compute :: inline (expr: $E, ret: *Tensor($T, $Loc)) #modify {
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

compute_simd :: inline (expr: $E, $loc := #caller_location) -> $RET #modify {
  RET = #dynamic_specialize Tensor(T = get_dtype(E), Loc = to_string(loc));
  return true;
} {
  shape := shape_of(expr);
  ret   := tensor(shape, #run get_dtype(E), loc);

  compute_simd(expr, *ret);

  return ret;
}

compute_simd :: inline (expr: $E, ret: *Tensor($T, $Loc)) #modify {
  T = get_dtype(E);
  return true;
} {
  #import "File";

  MAX_COST  :: #run get_cost(E);
  REG_COUNT :: #run min(MAX_COST + 1, 16);

  simd_n := ret.count & ~(cast(s64) 7);
  ret_ptr := ret.ptr;
  i := 0;

  declaration, tensors :: #run collect_pointer_declarations(E);
  #insert declaration;
  #run println("#########\n%", declaration);
  #insert #run read_entire_file("simd.jai");

  #asm AVX {
    ymm0:  vec === 0;  ymm1:  vec === 1;  ymm2:  vec === 2;  ymm3:  vec === 3;
    ymm4:  vec === 4;  ymm5:  vec === 5;  ymm6:  vec === 6;  ymm7:  vec === 7;
    ymm8:  vec === 8;  ymm9:  vec === 9;  ymm10: vec === 10; ymm11: vec === 11;
    ymm12: vec === 12; ymm13: vec === 13; ymm14: vec === 14; ymm15: vec === 15;
    ptr: gpr === 10;
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

eval_it :: inline (t: Tensor($T, $Loc), i: s64) -> T { 
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
