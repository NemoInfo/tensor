reg :: __reg;

eval_it_simd :: (v: *f32, out: reg, i: s64) #expand {
  #asm AVX { broadcastss out, [v]; }
}

eval_it_simd :: (t: *Tensor(f32), out: reg, i: s64) #expand {
  ptr := t.ptr;
  #asm AVX { movups out, [ptr + i * 8]; }
}

is_leaf :: ($T: Type) -> bool #compile_time {
  ti := cast(*Type_Info) T;
  if ti.type != .STRUCT return false;
  return (cast(*Type_Info_Struct) ti).name == "Tensor";
}

// ll * lr + r
eval_it_simd :: (expr: *Op2(.ADD, f32, Op2(.MUL, f32, $LL, $LR, $C2), $R, $C1), out: reg, i: s64) #expand {
  ll_cost :: #run get_cost(LL);
  lr_cost :: #run get_cost(LR);
  r_cost  :: #run get_cost(R);

  #if ll_cost >= lr_cost && ll_cost >= r_cost {
    eval_it_simd(*expr.lhs.lhs, out, i);
    #if #run is_leaf(LR) { // 132 lr ptr
      #asm AVX { r_reg: vec; }
      eval_it_simd(*expr.rhs, r_reg, i);
      ptr := expr.lhs.rhs.ptr;
      #asm AVX, FMA { fmadd132ps out, r_reg, [ptr + i * 8]; }
    } else #if #run is_leaf(R) { // 213 r ptr
      #asm AVX { lr_reg: vec; } eval_it_simd(*expr.lhs.rhs, lr_reg, i);
      ptr := expr.rhs.ptr;
      #asm AVX, FMA { fmadd213ps out, lr_reg, [ptr + i * 8]; }
    } else { // 213 nobody ptr
      #if lr_cost >= r_cost {
        #asm AVX {  r_reg: vec; } eval_it_simd(*expr.rhs, r_reg, i);
        #asm AVX { lr_reg: vec; } eval_it_simd(*expr.lhs.rhs, lr_reg, i);
      } else {
        #asm AVX { lr_reg: vec; } eval_it_simd(*expr.lhs.rhs, lr_reg, i);
        #asm AVX {  r_reg: vec; } eval_it_simd(*expr.rhs, r_reg, i);
      }
      #asm AVX, FMA { fmadd213ps out, lr_reg, r_reg; }
    }
  } else #if lr_cost >= ll_cost && lr_cost >= r_cost {
    eval_it_simd(*expr.lhs.rhs, out, i);
    #if #run is_leaf(LL) { // 132 ll ptr
      #asm AVX { r_reg: vec; }
      eval_it_simd(*expr.rhs, r_reg, i);
      ptr := expr.lhs.lhs.ptr;
      #asm AVX, FMA { fmadd132ps out, r_reg, [ptr + i * 8]; }
    } else #if #run is_leaf(R) { // 213 r ptr
      #asm AVX { ll_reg: vec; }
      eval_it_simd(*expr.lhs.lhs, ll_reg, i);
      ptr := expr.rhs.ptr;
      #asm AVX, FMA { fmadd213ps out, ll_reg, [ptr + i * 8]; }
    } else { // 213 nobody ptr
      #if ll_cost >= r_cost {
        #asm AVX { ll_reg: vec; } eval_it_simd(*expr.lhs.lhs, ll_reg, i);
        #asm AVX {  r_reg: vec; } eval_it_simd(*expr.rhs, r_reg, i);
      } else {
        #asm AVX {  r_reg: vec; } eval_it_simd(*expr.rhs, r_reg, i);
        #asm AVX { ll_reg: vec; } eval_it_simd(*expr.lhs.lhs, ll_reg, i);
      }
      #asm AVX, FMA { fmadd213ps out, ll_reg, r_reg; }
    }
  } else /* r_cost highest */ {
    eval_it_simd(*expr.rhs, out, i);
    #if #run is_leaf(LL) {
      #asm AVX { lr_reg: vec; }
      eval_it_simd(*expr.lhs.rhs, lr_reg, i);
      ptr := expr.lhs.lhs.ptr;
      #asm AVX, FMA { fmadd231ps out, lr_reg, [ptr + i * 8]; }
    } else #if #run is_leaf(LR) {
      #asm AVX { ll_reg: vec; }
      eval_it_simd(*expr.lhs.lhs, ll_reg, i);
      ptr := expr.lhs.rhs.ptr;
      #asm AVX, FMA { fmadd231ps out, ll_reg, [ptr + i * 8]; }
    } else {
      #if ll_cost >= r_cost {
        #asm AVX { ll_reg: vec; } eval_it_simd(*expr.lhs.lhs, ll_reg, i);
        #asm AVX { lr_reg: vec; } eval_it_simd(*expr.lhs.rhs, lr_reg, i);
      } else {
        #asm AVX { lr_reg: vec; } eval_it_simd(*expr.lhs.rhs, lr_reg, i);
        #asm AVX { ll_reg: vec; } eval_it_simd(*expr.lhs.lhs, ll_reg, i);
      }
      #asm AVX, FMA { fmadd231ps out, ll_reg, lr_reg; }
    }
  }
}

eval_it_simd :: (expr: *Op2($K, f32, $L, $R, $Cost), out: reg, i: s64, __dumy: $Dummy = void)
#expand {
  l_cost :: #run get_cost(L);
  r_cost :: #run get_cost(R);

  #if r_cost > l_cost {
    // heavy on right: eval right first into out
    eval_it_simd(*expr.rhs, out, i);

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
      eval_it_simd(*expr.lhs, lhs_reg, i);
      #if K == .ADD { #asm AVX { addps out, lhs_reg, out; } }
      #if K == .MUL { #asm AVX { mulps out, lhs_reg, out; } }
      #if K == .SUB { #asm AVX { subps out, lhs_reg, out; } }
      #if K == .DIV { #asm AVX { divps out, lhs_reg, out; } }
    }

  } else {
    // heavy on left (or equal): eval left first into out
    eval_it_simd(*expr.lhs, out, i);

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
      eval_it_simd(*expr.rhs, rhs_reg, i);
      #if K == .ADD { #asm AVX { addps out, out, rhs_reg; } }
      #if K == .MUL { #asm AVX { mulps out, out, rhs_reg; } }
      #if K == .SUB { #asm AVX { subps out, out, rhs_reg; } }
      #if K == .DIV { #asm AVX { divps out, out, rhs_reg; } }
    }
  }
}
