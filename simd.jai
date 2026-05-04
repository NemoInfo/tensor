reg :: __reg;

eval_it_simd :: (t: Tensor(f32), out: reg, i: s64) #expand {
  // #run dbg("mv tensor");
  ptr := t.ptr;
  #asm AVX {
    movups out, [ptr + i * 8];
  }
}

is_leaf :: ($T: Type) -> bool #compile_time {
  ti := cast(*Type_Info) T;
  if ti.type != .STRUCT return false;
  return (cast(*Type_Info_Struct) ti).name == "Tensor";
}

eval_it_simd :: (expr: Op2($K, f32, $L, $R, $Cost), out: reg, i: s64) #expand {
  l_cost :: #run get_cost(L);
  r_cost :: #run get_cost(R);

  #if r_cost > l_cost {
    // heavy on right: eval right first into out
    eval_it_simd(expr.rhs, out, i);

    #if #run is_leaf(L) {
      // left is a tensor: fold as memory operand
      // ADD/MUL are commutative so we can flip; SUB/DIV need a tmp
      ptr := expr.lhs.ptr;
      #if K == .ADD { #asm AVX { addps out, out, [ptr + i * 8]; } }
      #if K == .MUL { #asm AVX { mulps out, out, [ptr + i * 8]; } }
      #if K == .SUB { #asm AVX { tmp: vec;
                                 movups tmp, [ptr + i * 8];
                                 subps  tmp, tmp, out;
                                 movaps out, tmp; } }
      #if K == .DIV { #asm AVX { tmp: vec;
                                 movups tmp, [ptr + i * 8];
                                 divps  tmp, tmp, out;
                                 movaps out, tmp; } }
    } else {
      // left is a subtree: needs its own register
      #asm AVX { lhs_reg: vec; }
      eval_it_simd(expr.lhs, lhs_reg, i);
      #if K == .ADD { #asm AVX { addps out, lhs_reg, out; } }
      #if K == .MUL { #asm AVX { mulps out, lhs_reg, out; } }
      #if K == .SUB { #asm AVX { subps out, lhs_reg, out; } }
      #if K == .DIV { #asm AVX { divps out, lhs_reg, out; } }
    }

  } else {
    // heavy on left (or equal): eval left first into out
    eval_it_simd(expr.lhs, out, i);

    #if #run is_leaf(R) {
      // #run dbg("left heavy, right leaf");
      // right is a tensor: fold as memory operand
      ptr := expr.rhs.ptr;
      #if K == .ADD { #asm AVX { addps out, out, [ptr + i * 8]; } }
      #if K == .MUL { #asm AVX { mulps out, out, [ptr + i * 8]; } }
      #if K == .SUB { #asm AVX { subps out, out, [ptr + i * 8]; } }
      #if K == .DIV { #asm AVX { divps out, out, [ptr + i * 8]; } }
    } else {
      // right is a subtree: needs its own register
      #asm AVX { rhs_reg: vec; }
      eval_it_simd(expr.rhs, rhs_reg, i);
      #if K == .ADD { #asm AVX { addps out, out, rhs_reg; } }
      #if K == .MUL { #asm AVX { mulps out, out, rhs_reg; } }
      #if K == .SUB { #asm AVX { subps out, out, rhs_reg; } }
      #if K == .DIV { #asm AVX { divps out, out, rhs_reg; } }
    }
  }
}
