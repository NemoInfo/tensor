Thing :: struct {
  x:  s64;
  xs: [] s64;
}

comp_time_thing :: () -> Thing #compile_time {
  xs: [] s64;
  array_resize(*xs, xs.count + 1);
  xs.data[xs.count - 1] = 420;
  array_resize(*xs, xs.count + 1);
  xs.data[xs.count - 1] = 420;
  return .{x = 69, xs = xs};
}

comp_time_xs :: () -> [..] s64 #compile_time {
  xs: [..] s64;
  array_add(*xs, 420);
  return xs;
}

main :: () {
  X :: #run comp_time_xs();
  THING :: #run comp_time_thing();
  print("runtime: %\n", X);
  print("runtime: %\n", THING);
}

#import "Basic";
