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

Leaf :: struct($T: Type) { tensor: *Tensor(T); }

Op1 :: struct($K: OpKind, $T: Type, $C: Type) {
  arg: C;
}

Op2 :: struct($K: OpKind, $T: Type, $L: Type, $R: Type) {
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
%1 :: (l: $L, r: $R) -> Op2(.%2, $T, L, R) #modify {
  l_dtype := get_dtype(L);
  r_dtype := get_dtype(R);
  if l_dtype != r_dtype return false, tprint("lhs dtype != rhs dtype mismatch \% != \%", l_dtype, r_dtype);
  T = l_dtype;
  return true; 
} { return .{l, r}; }
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
}

get_dtype :: (S: Type) -> Type #compile_time {
  ti := cast(*Type_Info) S;
  if ti.type != .STRUCT return S;
  tis := cast(*Type_Info_Struct) ti;
  if tis.name == {
    case "Leaf"; #through;
    case "Op1";  #through;
    case "Op2";
    for tis.specified_parameters {
      if it.name != "T" continue;
      return (cast(*Type) (tis.constant_storage.data + it.offset_into_constant_storage)).*;
    }
  }
  return S;
}

eval_it :: inline (val: $T, nd_index: []s64) -> T #modify { return T == get_dtype(T); } { return val; }

eval_it :: inline (using leaf: Leaf($T), nd_index: []s64) -> T {
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

eval_it :: inline (expr: Op2($K, $T, $L, $R), nd_index: []s64) -> T {
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

shape_of :: (expr: Op2($K, $T, $L, $R)) -> []s64 {
  return broadcast_shape(shape_of(expr.lhs), shape_of(expr.rhs));
}
shape_of :: (expr: Op1($K, $T, $C))     -> []s64 { return shape_of(expr.arg); }
shape_of :: (using leaf: Leaf($T))      -> []s64 { return tensor.shape; }
shape_of :: (expr: $T) -> []s64 #modify {
  if T == get_dtype(T) return true;
  return false;
} { return .[]; }

compute :: inline (expr: $E) -> Tensor($T) #modify {
  T = get_dtype(E);
  return true;
} {
  ret      := tensor(shape_of(expr), T);
  nd_index := NewArray(ret.ndim, s64,, temp);
  for 0..ret.count-1 {
    tmp := it;
    for < 0..ret.ndim-1 {
      nd_index[it] = tmp % ret.shape[it];
      tmp /= ret.shape[it];
    }
    ret.data[it] = eval_it(expr, nd_index);
  }
  return ret;
}
