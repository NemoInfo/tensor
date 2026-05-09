ymm :: (reg_i: s64) -> string #compile_time {
  return tprint("ymm%", reg_i);
}

get_tensor_id :: (path: [] u8) -> s64 #compile_time {
  tensord_id := -1;
  for tensors if array_eq(it.index, path) tensord_id = it_index;
  if tensord_id == -1  panic("Could not resolve tensor path %", path);
  return tensord_id;
}

t :: (path: [] u8) -> mov: string, reg: string  #compile_time {
  tensord_id := get_tensor_id(path);
  if tensors.count > 11 && tensord_id >= 10 { // tensor is in ts
    return tprint("#asm { mov ptr, [ts_ptr + %]; }", (tensord_id - 10) * size_of(*void)), "[ptr + i]";
  } else { // tensor is in t<tensord_id>
    return "", tprint("[t% + i]", tensord_id);
  }
}

eval_it_simd :: (v: *f32, $next_reg : s64, $path := u8.[]) #expand {
  #insert #run tprint("#asm AVX { broadcastss.y %1, [v]; }", ymm(next_reg));
}

eval_it_simd :: (t_ptr: *Tensor(f32), $next_reg : s64, $path := u8.[]) #expand {
  mov, reg :: #run t(path);
  #insert mov;
  #insert #run tprint("#asm AVX { movups.y %1, %2; }", ymm(next_reg), reg);
}

is_leaf :: ($T: Type) -> bool #compile_time {
  ti := cast(*Type_Info) T;
  if ti.type != .STRUCT return false;
  return (cast(*Type_Info_Struct) ti).name == "Tensor";
}

eval_it_simd :: (expr: *Op2($K, f32, $L, $R, $Cost), $next_reg : s64, 
                 $path := u8.[], __dumy: $Dummy = void)
#expand {
  l_cost :: #run get_cost(L);
  r_cost :: #run get_cost(R);

  #if r_cost > l_cost { // heavy on right: eval right first into out
    eval_it_simd(*expr.rhs, next_reg, #run rhs(path));
    #if (K == .ADD || K == .MUL) && #run is_leaf(L) {
      // ADD/MUL are commutative so we can flip; SUB/DIV need a register
      mov, reg :: #run t(lhs(path));
      #insert mov;
      #insert #run tprint(#string JAI
        #if K == .ADD #asm AVX { addps.y %1, %1, %2; }
        #if K == .MUL #asm AVX { mulps.y %1, %1, %2; }
      JAI, ymm(next_reg), reg);
    } else { // left is a subtree: needs its own register
      eval_it_simd(*expr.lhs, #run next_reg + 1, #run lhs(path));
      #insert #run tprint(#string JAI
        #if K == .ADD #asm AVX { addps.y %1, %2, %1; }
        #if K == .MUL #asm AVX { mulps.y %1, %2, %1; }
        #if K == .SUB #asm AVX { subps.y %1, %2, %1; }
        #if K == .DIV #asm AVX { divps.y %1, %2, %1; }
      JAI, ymm(next_reg), ymm(next_reg + 1));
    }
  } else { // heavy on left (or equal): eval left first into out
    eval_it_simd(*expr.lhs, next_reg, #run lhs(path));
    #if #run is_leaf(R) { // right is a tensor: fold as memory operand
      mov, arg2 :: #run t(rhs(path));
      #insert mov;
    } else { // right is a subtree: needs its own register
      eval_it_simd(*expr.rhs, #run next_reg + 1, #run rhs(path));
      arg2 :: #run ymm(next_reg + 1);
    }
    #insert #run tprint(#string JAI
      #if K == .ADD #asm AVX { addps.y %1, %1, %2; }
      #if K == .MUL #asm AVX { mulps.y %1, %1, %2; }
      #if K == .SUB #asm AVX { subps.y %1, %1, %2; }
      #if K == .DIV #asm AVX { divps.y %1, %1, %2; }
    JAI, ymm(next_reg), arg2);
  }
}

// eval_it_simd :: (expr: *Op2(.ADD, f32, Op2(.MUL, f32, $LL, $LR, $C2), $R, $C1), 
//                  $next_reg : s64, $path := u8.[]) #expand {
//   ll_cost :: #run get_cost(LL);
//   lr_cost :: #run get_cost(LR);
//   r_cost  :: #run get_cost(R);
// 
//   fmadd :: (var: s64, a: string, b: string, c: string) -> string {
//     return tprint("#asm AVX, FMA { fmadd%1ps %2, %3, %4; }", var, a, b, c);
//   }
// 
//   #if ll_cost >= lr_cost && ll_cost >= r_cost {
//     eval_it_simd(*expr.lhs.lhs, next_reg, #run lhs(lhs(path)));
//     #if #run is_leaf(LR) { // 132 lr ptr
//       eval_it_simd(*expr.rhs, #run next_reg + 1, #run rhs(path));
//       #insert #run fmadd(132, ymm(next_reg), ymm(next_reg + 1), t(rhs(lhs(path))));
//     } else #if #run is_leaf(R) { // 213 r ptr
//       eval_it_simd(*expr.lhs.rhs, #run next_reg + 1, #run rhs(lhs(path)));
//       #insert #run fmadd(213, ymm(next_reg), ymm(next_reg + 1), t(rhs(path)));
//     } else { // 213 nobody ptr
//       #if lr_cost >= r_cost {
//         eval_it_simd(*expr.rhs,     #run next_reg + 1, #run rhs(path));
//         eval_it_simd(*expr.lhs.rhs, #run next_reg + 2, #run rhs(lhs(path)));
//         #insert #run fmadd(213, ymm(next_reg), ymm(next_reg + 2), ymm(next_reg + 1));
//       } else {
//         eval_it_simd(*expr.lhs.rhs, #run next_reg + 1, #run rhs(lhs(path)));
//         eval_it_simd(*expr.rhs,     #run next_reg + 2, #run rhs(path));
//         #insert #run fmadd(213, ymm(next_reg), ymm(next_reg + 1), ymm(next_reg + 2));
//       }
//     }
//   } else #if lr_cost >= ll_cost && lr_cost >= r_cost {
//     eval_it_simd(*expr.lhs.rhs, next_reg, #run rhs(lhs(path)));
//     #if #run is_leaf(LL) { // 132 ll ptr
//       eval_it_simd(*expr.rhs, #run next_reg + 1, #run rhs(path));
//       #insert #run fmadd(132, ymm(next_reg), ymm(next_reg + 1), t(lhs(lhs(path))));
//     } else #if #run is_leaf(R) { // 213 r ptr
//       eval_it_simd(*expr.lhs.lhs, #run next_reg + 1, #run lhs(lhs(path)));
//       fma(213, ymm(next_reg), ymm(next_reg + 1), t(rhs));
//     } else { // 213 nobody ptr
//       #if ll_cost >= r_cost {
//         eval_it_simd(*expr.lhs.lhs, #run next_reg + 1, #run lhs(lhs(path)));
//         eval_it_simd(*expr.rhs, #run next_reg + 2, #run rhs(path));
//         fma(213, ymm(next_reg), ymm(next_reg + 1), ymm(next_reg + 2));
//       } else {
//         eval_it_simd(*expr.rhs, #run next_reg + 1, #run rhs(path));
//         eval_it_simd(*expr.lhs.lhs, #run next_reg + 2, #run lhs(lhs(path)));
//         fma(213, ymm(next_reg), ymm(next_reg + 2), ymm(next_reg + 1));
//       }
//     }
//   } else /* r_cost highest */ {
//     eval_it_simd(*expr.rhs, next_reg, #run rhs(path));
//     #if #run is_leaf(LL) {
//       eval_it_simd(*expr.lhs.rhs, #run next_reg + 1, #run rhs(lhs(path)));
//       #insert #run fmadd(231, ymm(next_reg), ymm(next_reg + 1), t(lhs(lhs)));
//     } else #if #run is_leaf(LR) {
//       eval_it_simd(*expr.lhs.lhs, #run next_reg + 1, #run lhs(lhs(path)));
//       #insert #run fmadd(231, ymm(next_reg), ymm(next_reg + 1), t(rhs(lhs)));
//     } else {
//       #if ll_cost >= r_cost {
//         eval_it_simd(*expr.lhs.lhs, #run next_reg + 1, #run lhs(lhs(path)));
//         eval_it_simd(*expr.lhs.rhs, #run next_reg + 2, #run rhs(lhs(path)));
//         #insert #run fmadd(231, ymm(next_reg), ymm(next_reg + 1), ymm(next_reg + 2));
//       } else {
//         eval_it_simd(*expr.lhs.rhs, #run next_reg + 1, #run rhs(lhs(path)));
//         eval_it_simd(*expr.lhs.lhs, #run next_reg + 2, #run lhs(lhs(path)));
//         #insert #run fmadd(231, ymm(next_reg), ymm(next_reg + 2), ymm(next_reg + 1));
//       }
//     }
//   }
// }
