reg :: __reg;

ymm :: (reg_i: s64) -> string #compile_time {
  return tprint("ymm%", reg_i);
}

eval_it_simd :: (v: *f32, $next_reg : s64, $path: string = "") #expand {
  #insert #run tprint("#asm AVX { broadcastss.y %1, [v]; }", ymm(next_reg));
}

eval_it_simd :: (t: *Tensor(f32), $next_reg : s64, $path: string = "") #expand {
  vn  :: #run find_vn(tprint("%.ptr", path));
  #insert #run tprint("#asm AVX { movups.y %1, [v%2 + i]; }", ymm(next_reg), vn);
}

is_leaf :: ($T: Type) -> bool #compile_time {
  ti := cast(*Type_Info) T;
  if ti.type != .STRUCT return false;
  return (cast(*Type_Info_Struct) ti).name == "Tensor";
}

eval_it_simd :: (expr: *Op2($K, f32, $L, $R, $Cost), $next_reg : s64, 
                 $path: string = "", __dumy: $Dummy = void)
#expand {
  l_cost :: #run get_cost(L);
  r_cost :: #run get_cost(R);

  #if r_cost > l_cost {
    // heavy on right: eval right first into out
    eval_it_simd(*expr.rhs, next_reg, #run tprint("%.rhs", path));

    #if (K == .ADD || K == .MUL) && #run is_leaf(L) {
      // ADD/MUL are commutative so we can flip; SUB/DIV need a register
      vn :: #run find_vn(tprint("%.lhs.ptr", path));
      #insert #run tprint(#string JAI
        #if K == .ADD #asm AVX { addps.y %1, %1, [v%2 + i]; }
        #if K == .MUL #asm AVX { mulps.y %1, %1, [v%2 + i]; }
      JAI, ymm(next_reg), vn);
    } else { // left is a subtree: needs its own register
      eval_it_simd(*expr.lhs, #run next_reg + 1, #run tprint("%.lhs", path));
      #insert #run tprint(#string JAI
        #if K == .ADD #asm AVX { addps.y %1, %2, %1; }
        #if K == .MUL #asm AVX { mulps.y %1, %2, %1; }
        #if K == .SUB #asm AVX { subps.y %1, %2, %1; }
        #if K == .DIV #asm AVX { divps.y %1, %2, %1; }
      JAI, ymm(next_reg), ymm(next_reg + 1));
    }

  } else {
    // heavy on left (or equal): eval left first into out
    eval_it_simd(*expr.lhs, next_reg, #run tprint("%.lhs", path));

    #if #run is_leaf(R) {
      // right is a tensor: fold as memory operand
      vn :: #run find_vn(tprint("%.rhs.ptr", path));
      #insert #run tprint(#string JAI
        #if K == .ADD #asm AVX { addps.y %1, %1, [v%2 + i]; }
        #if K == .MUL #asm AVX { mulps.y %1, %1, [v%2 + i]; }
        #if K == .SUB #asm AVX { subps.y %1, %1, [v%2 + i]; }
        #if K == .DIV #asm AVX { divps.y %1, %1, [v%2 + i]; }
      JAI, ymm(next_reg), vn);
    } else {
      // right is a subtree: needs its own register
      eval_it_simd(*expr.rhs, #run next_reg + 1, #run tprint("%.rhs", path));
      #insert #run tprint(#string JAI
      #if K == .ADD #asm AVX { addps.y %1, %1, %2; }
      #if K == .MUL #asm AVX { mulps.y %1, %1, %2; }
      #if K == .SUB #asm AVX { subps.y %1, %1, %2; }
      #if K == .DIV #asm AVX { divps.y %1, %1, %2; }
      JAI, ymm(next_reg), ymm(next_reg + 1));
    }
  }
}

find_vn :: (path: string) -> s64  #compile_time {
  for tensors if it == path return it_index;
  return -1;
}
