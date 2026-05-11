Thing :: struct($Loc: Source_Code_Location = .{}) {data: s64;}

thing :: (val: s64, $loc := #caller_location) -> $RET #modify {
  RET = #dynamic_specialize Thing(Loc = loc);
  return true;
} { return .{val}; }

thing_thing :: (val: s64, $loc := #caller_location) -> $RET #modify {
  RET = #dynamic_specialize Thing(Loc = loc);
  return true;
} {
  return thing(val, loc);
}

main :: () {
  t1 := thing_thing(1);
  t2 := thing_thing(2);
  t3 := thing_thing(3);
  print("%\n", type_of(t1));
  print("%\n", type_of(t2));
  print("%\n", type_of(t3));
  print("%\n", size_of(Thing));
  
  print("%\n", size_of(Source_Code_Location));
}

#import "Basic";
