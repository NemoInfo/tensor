f32 :: float32;
f64 :: float64;

transmute :: inline ($toT: Type, x: $T) -> toT {
    union { from : T = ---; to : toT = ---; }    
    from = x;
    return to;
}

println :: inline (s: string, args: ..Any) {
  #import "Basic";
  print(s, ..args); print("\n");
}

dbg :: (a: Code, loc := #caller_location) #expand {
    val := #insert a;
    str :: #run print_code(a); 
    print("[%:%:%] % = %\n", loc.fully_pathed_filename, loc.line_number, loc.character_number, str, val);
}

print_code :: (a: Code) -> string #compile_time {
  #import "Compiler";
  #import "Program_Print";
  root, expr :=  compiler_get_nodes(a);
  sb: String_Builder;
  print_expression(*sb, root);
  return builder_to_string(*sb);
}

eprint :: inline (s: string, args: ..Any) {
  write_string(tprint(s, ..args), to_standard_error=true);
}

eprintln :: inline (s: string, args: ..Any) {
  eprint(s, ..args); eprint("\n");
}

panic :: inline (s: string, args: ..Any, exit_code: s32 = 1, loc := #caller_location) {
  #import "Print_Color";
  eprint("[%:%:%] ", loc.fully_pathed_filename, loc.line_number, loc.character_number);
  print_color("panicked:", color=.RED, .BOLD, to_standard_error=true);
  eprintln(" %", tprint(s, ..args));
  exit(exit_code);
}

center_pad :: (str: string, width: int, padding_char: u8 = #char " ") -> string {
  if str.count >= width return str;

  padding_needed := width - str.count;
  left_padding := padding_needed / 2;
  right_padding := padding_needed - left_padding;

  lp : [] u8 = NewArray(left_padding,  u8); defer array_free(lp);
  rp : [] u8 = NewArray(right_padding, u8); defer array_free(rp);
  for *lp it.* = padding_char;
  for *rp it.* = padding_char;

  return tprint("%%%", cast(string) lp, str, cast(string) rp);
}
