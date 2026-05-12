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
