NodeId :: s64;
Node :: struct {
  kind: OpKind = xx 0;
  lhs:  NodeId = -1;
  rhs:  NodeId = -1;
  loc:  string = "";
  #overlay (lhs)  arg: NodeId;
  #overlay (kind) key: [size_of(string) + size_of(NodeId) * 2 + size_of(OpKind)] u8;
  suffix: string = "";
  crit_path: s64;
  parents: [] NodeId;
}

Operand :: struct {
  node:      NodeId;
  reg:       s64;      // current YMM register (‑1 if not in reg)
  slot:      s64;      // spill slot (‑1 if not spilled)
  next_use:  s64;      // next schedule index where needed (or MAX_S64)
  use_idx:   s64;      // index into use_positions list
}

is_op2    :: inline (using node: Node) -> bool { return lhs != -1 && rhs != -1; }
is_op1    :: inline (using node: Node) -> bool { return arg != -1 && rhs == -1; }
is_tensor :: inline (using node: Node) -> bool { return arg == -1 && rhs == -1; }

schedule_dag :: (nodes: [] Node) -> [] NodeId #compile_time {
  remaining_children := NewArray(nodes.count, s64);
  for nodes 
    if   is_tensor(it) remaining_children[it_index] = 0;
    else if is_op1(it) remaining_children[it_index] = 1;
    else if is_op2(it) remaining_children[it_index] = 2;

  ready : [..] NodeId;
  for nodes if remaining_children[it_index] == 0
    array_add(*ready, it_index);

  scheduled : [..] NodeId;

  while ready.count > 0 {
    best_idx  := -1;
    best_crit := -1;
    for node_id: ready {
      crit := nodes[node_id].crit_path;
      if crit > best_crit {
        best_crit = crit;
        best_idx = it_index;
      }
    }
    node_id := ready[best_idx];
    array_unordered_remove_by_index(*ready, best_idx);
    array_add(*scheduled, node_id);

    for parent_id: nodes[node_id].parents {
      remaining_children[parent_id] -= 1;
      if remaining_children[parent_id] == 0
        array_add(*ready, parent_id);
    }
  }

  return scheduled;
}

build_dag :: (E: Type) -> (root_id: NodeId, nodes: [] Node) {
  nodes: [..] Node; 
  id := build_dag(E, *nodes);
  return id, nodes;
}

nodes_find :: (nodes: [..] Node, item: Node) -> bool, s64 {
  for nodes if array_eq(it.key, item.key) return true, it_index;
  return false, -1;
}

nodes_add_if_unique :: (nodes: *[..] Node, item: Node) -> bool, s64 {
  found, index := nodes_find(nodes.*, item);   
  if found return false, index;
  array_add(nodes, item);
  return true, nodes.count-1;
}

build_dag :: (E: Type, nodes: *[..] Node, suffix := "") -> NodeId {
  ti := cast(*Type_Info) E;
  if ti.type != .STRUCT  return -1;
  tis := cast(*Type_Info_Struct) ti;
  if tis.name == {
    case "Tensor"; for tis.specified_parameters if it.name == "Loc" {
        loc : string = (cast(*string) (tis.constant_storage.data + it.offset_into_constant_storage)).*;
        _, id := nodes_add_if_unique(nodes, .{loc = loc, suffix = ptr(suffix)});
        return id;
      }
    case "Op1";
      kind: OpKind; 
      for tis.specified_parameters if it.name == "K"
        kind = (cast(*OpKind) (tis.constant_storage.data + it.offset_into_constant_storage)).*;
      child_type : Type;
      for tis.specified_parameters if it.name == "C"
        child_type = (cast(*Type) (tis.constant_storage.data + it.offset_into_constant_storage)).*;
      arg_id := build_dag(child_type, nodes, arg(suffix));
      crit_path := 1 + nodes.*[arg_id].crit_path;
      added, id := nodes_add_if_unique(nodes, .{kind = kind, arg = arg_id, crit_path = crit_path});
      if !added return id;

      arg_parents := *nodes.*[arg_id].parents; 
      array_resize(arg_parents, arg_parents.count + 1);
      arg_parents.*[arg_parents.count - 1] = id;
      return id;
    case "Op2"; 
      l_type, r_type : Type;
      kind: OpKind; 
      for tis.specified_parameters if it.name == "L"
        l_type = (cast(*Type) (tis.constant_storage.data + it.offset_into_constant_storage)).*;
      for tis.specified_parameters if it.name == "R"
        r_type = (cast(*Type) (tis.constant_storage.data + it.offset_into_constant_storage)).*;
      for tis.specified_parameters if it.name == "K"
        kind   = (cast(*OpKind) (tis.constant_storage.data + it.offset_into_constant_storage)).*;
      lhs_id := build_dag(l_type, nodes, lhs(suffix));
      rhs_id := build_dag(r_type, nodes, rhs(suffix));
      crit_path := 1 + max(nodes.*[lhs_id].crit_path, nodes.*[rhs_id].crit_path);

      added, id := nodes_add_if_unique(nodes, .{kind = kind, lhs = lhs_id, rhs = rhs_id, crit_path = crit_path});
      if !added return id;

      lhs_parents := *nodes.*[lhs_id].parents; 
      array_resize(lhs_parents, lhs_parents.count + 1);
      lhs_parents.*[lhs_parents.count - 1] = id;

      rhs_parents := *nodes.*[rhs_id].parents; 
      array_resize(rhs_parents, rhs_parents.count + 1);
      rhs_parents.*[rhs_parents.count - 1] = id;

      return id;
  }
  return -1;
}

binary_op_to_string :: (k: OpKind) -> string #compile_time {
  if k == .ADD return "add";
  if k == .SUB return "sub";
  if k == .MUL return "mul";
  if k == .DIV return "div";
  panic("unsuported");
  return "???";
}

SimdCodeResult :: struct {
  loop_body: string;
  pre_decl: string;
  root_reg: s64;
}

generate_simd_code :: (E: Type) -> (loop_body: string, pre_decl: string, root_reg: s64) #compile_time {
  declaration, tensors := collect_pointer_declarations(E);
  root_id, nodes := build_dag(E);
  schedule       := schedule_dag(nodes);
  // println("---   nodes  ---");
  // for nodes println("%: % ", it_index, it);
  // println("--- schedule ---\n%", schedule);
  REG_COUNT :: 15; // leave ymm15 for spilling/broadcast

  // 2. Use positions
  use_positions : [][..] s64 = NewArray(nodes.count, [..]s64);
  for sched_idx : 0..schedule.count-1 {
    p_id := schedule[sched_idx];
    p := nodes[p_id];
    if p.lhs != -1 { array_add(*use_positions[p.lhs], sched_idx); }
    if p.rhs != -1 { array_add(*use_positions[p.rhs], sched_idx); }
  }

  op_state : [] Operand = NewArray(nodes.count, Operand);
  for id : 0..nodes.count-1 {
    op_state[id].node     = id;
    op_state[id].reg      = -1;
    op_state[id].slot     = -1;
    op_state[id].use_idx  = 0;
    op_state[id].next_use = MAX_S64;
  }

  reg_owner : [REG_COUNT] s64; 
  for *reg_owner it.* = -1;

  spill_count : s64 = 0;
  sb : String_Builder;

  find_spill_victim :: (reg_owner: [] s64, op_state: [] Operand) -> s64 {
    victim   := -1;
    max_dist := -1;
    for reg_id, node_id: reg_owner {
      if node_id == -1 continue;
      if op_state[node_id].next_use > max_dist {
        max_dist = op_state[node_id].next_use;
        victim = reg_id;
      }
    }
    return victim;
  }

  for schedule_it, cur_id: schedule {
    cur_node := nodes[cur_id];

    // ---- Leaf tensor: load it and continue ----
    if is_tensor(cur_node) {
      found, r := array_find(reg_owner, -1);   // find free register
      if !found {
        r = find_spill_victim(reg_owner, op_state);
        victim_id := reg_owner[r];
        victim_op := *op_state[victim_id];
        if victim_op.next_use != MAX_S64 {
          if victim_op.slot == -1 { victim_op.slot = spill_count; spill_count += 1; }
          print_to_builder(*sb,
          "#asm AVX { movups.y [spill_buf + % * 32], ymm%; }\n",
          victim_op.slot, r);
        }
        reg_owner[r] = -1;
        op_state[victim_id].reg = -1;
      }

      // Load the leaf into r
      foundd, t_info := table_find(*tensors, cur_node.loc); assert(foundd);
      print_to_builder(*sb, "#asm AVX { movups.y ymm%, [t% + i]; }\n", r, t_info.t_id);

      reg_owner[r] = cur_id;
      op_state[cur_id].reg = r;
      if use_positions[cur_id].count > 0 {
        op_state[cur_id].next_use = use_positions[cur_id][0];
      } else {
        op_state[cur_id].next_use = MAX_S64;
      }
      op_state[cur_id].use_idx = 0;
      continue;   // done with this node
    }

    // ---- Non‑leaf operation (Op1 / Op2) from here on ----
    children : [..] s64;
    if cur_node.lhs != -1 { array_add(*children, cur_node.lhs); }
    if cur_node.rhs != -1 { array_add(*children, cur_node.rhs); }

    // Bring children into registers
    for child_id: children {
      child_op := *op_state[child_id];
      if child_op.reg != -1  continue;   // already in register

      // Allocate register
      found, r := array_find(reg_owner, -1);
      if !found {
        r = find_spill_victim(reg_owner, op_state);
        victim_id := reg_owner[r];
        victim_op := *op_state[victim_id];
        if victim_op.next_use != MAX_S64 {
          if victim_op.slot == -1 { victim_op.slot = spill_count; spill_count += 1; }
          print_to_builder(*sb,
          "#asm AVX { movups.y [spill_buf + % * 32], ymm%; }\n",
          victim_op.slot, r);
        }
        reg_owner[r] = -1;
        op_state[victim_id].reg = -1;
      }

      // Load child into r
      if child_op.slot != -1 {
        print_to_builder(*sb,
        "#asm AVX { movups.y ymm%, [spill_buf + % * 32]; }\n",
        r, child_op.slot);
      } else {
        found, t_info := table_find(*tensors, nodes[child_id].loc);
        assert(found);
        print_to_builder(*sb,
        "#asm AVX { movups.y ymm%, [t% + i]; }\n",
        r, t_info.t_id);
      }
      reg_owner[r] = child_id;
      op_state[child_id].reg = r;
    }

    // Decide result register (reuse dead operand if possible)
    res_reg := -1;
    maybe_reuse := -1;
    for child_id: children {
      child_op := *op_state[child_id];
      if child_op.next_use == schedule_it {   // this is the current use
        if child_op.use_idx + 1 >= use_positions[child_id].count {
          maybe_reuse = child_op.reg;
          break;
        }
      }
    }
    if maybe_reuse != -1 {
      res_reg = maybe_reuse;
    } else {
      found, r := array_find(reg_owner, -1);
      if !found {
        r = find_spill_victim(reg_owner, op_state);
        victim_id := reg_owner[r];
        victim_op := *op_state[victim_id];
        if victim_op.next_use != MAX_S64 {
          if victim_op.slot == -1 { victim_op.slot = spill_count; spill_count += 1; }
          print_to_builder(*sb,
          "#asm AVX { movups.y [spill_buf + % * 32], ymm%; }\n",
          victim_op.slot, r);
        }
        reg_owner[r] = -1;
        op_state[victim_id].reg = -1;
      }
      res_reg = r;
    }

    // Emit arithmetic
    if is_op1(cur_node) {
      arg_reg := op_state[cur_node.arg].reg;
      panic("todo");
      // op_str  := unary_op_to_string(cur_node.kind);
      // print_to_builder(*sb,
      // "#asm AVX { v%ps.y ymm%, ymm%; }\n",
      // op_str, res_reg, arg_reg);
    } else {
      lhs_reg := op_state[cur_node.lhs].reg;
      rhs_reg := op_state[cur_node.rhs].reg;
      op_str  := binary_op_to_string(cur_node.kind);
      print_to_builder(*sb,
      "#asm AVX { %ps.y ymm%, ymm%, ymm%; }\n",
      op_str, res_reg, lhs_reg, rhs_reg);
    }

    for child_id: children {
      child_op := *op_state[child_id];
      child_op.use_idx += 1;
      op_state[child_id].use_idx = child_op.use_idx;
      if child_op.use_idx < use_positions[child_id].count
        op_state[child_id].next_use = use_positions[child_id][child_op.use_idx];
      else
        op_state[child_id].next_use = MAX_S64;
      if child_op.next_use == MAX_S64 && child_op.reg != res_reg {
        reg_owner[child_op.reg] = -1;
        op_state[child_id].reg = -1;
      }
    }

    // Assign result to current node
    reg_owner[res_reg] = cur_id;
    op_state[cur_id].reg = res_reg;
    if use_positions[cur_id].count > 0 {
      op_state[cur_id].next_use = use_positions[cur_id][0];
    } else {
      op_state[cur_id].next_use = MAX_S64;
    }
    op_state[cur_id].use_idx = 0;
  }

  // Spill declaration
  pre_decl : string = declaration;
  if spill_count > 0 {
    pre_decl = tprint("spill_buf : [%] f32 = ---;\n%", spill_count * 8, pre_decl);
  }

  return builder_to_string(*sb), pre_decl, op_state[root_id].reg;
}
