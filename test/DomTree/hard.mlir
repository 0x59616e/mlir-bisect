// RUN: %mlir-bisect %s --dump-idom | %FileCheck %s

// CHECK: [[ARG0:<block argument> of type 'tensor<f32>' at index: 0]]
// CHECK-NEXT: [[A:%[0-9]+]] = "test.A"
// CHECK-NEXT: [[B:%[0-9]+]] = "test.B"
// CHECK-NEXT: [[C:%[0-9]+]] = "test.C"
// CHECK-NEXT: [[D:%[0-9]+]] = "test.D"
// CHECK-NEXT: [[E:%[0-9]+]] = "test.E"
// CHECK-NEXT: [[F:%[0-9]+]] = "test.F"
// CHECK-NEXT: [[G:%[0-9]+]] = "test.G"
// CHECK-NEXT: [[H:%[0-9]+]] = "test.H"
// CHECK-NEXT: [[I:%[0-9]+]] = "test.I"
// CHECK-NEXT: [[J:%[0-9]+]] = "test.J"
// CHECK-NEXT: [[K:%[0-9]+]] = "test.K"
// CHECK-NEXT: [[L:%[0-9]+]] = "test.L"
// CHECK-NEXT: [[M:%[0-9]+]] = "test.M"
// CHECK-NEXT: [[N:%[0-9]+]] = "test.N"
// CHECK-NEXT: [[O:%[0-9]+]] = "test.O"
// CHECK-NEXT: ---
// CHECK-NEXT: <<NULL VALUE>> => [[ARG0]]
// CHECK-NEXT: [[ARG0]] => [[A]]
// CHECK-NEXT: [[ARG0]] => [[B]]
// CHECK-NEXT: [[ARG0]] => [[C]]
// CHECK-NEXT: [[A]]{{.*}} => [[D]]
// CHECK-NEXT: [[ARG0]] => [[E]]
// CHECK-NEXT: [[B]]{{.*}} => [[F]]
// CHECK-NEXT: [[C]]{{.*}} => [[G]]
// CHECK-NEXT: [[E]]{{.*}} => [[H]]
// CHECK-NEXT: [[ARG0]] => [[I]]
// CHECK-NEXT: [[ARG0]] => [[J]]
// CHECK-NEXT: [[ARG0]] => [[K]]
// CHECK-NEXT: [[ARG0]] => [[L]]
// CHECK-NEXT: [[K]]{{.*}} => [[M]]
// CHECK-NEXT: [[L]]{{.*}} => [[N]]
// CHECK-NEXT: [[ARG0]] => [[O]]

func.func @hard(%arg0: tensor<f32>) -> tensor<f32> {
  %A = "test.A"(%arg0) : (tensor<f32>) -> tensor<f32>
  %B = "test.B"(%A, %arg0, %arg0) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  %C = "test.C"(%arg0) : (tensor<f32>) -> tensor<f32>
  %D = "test.D"(%A) : (tensor<f32>) -> tensor<f32>
  %E = "test.E"(%D, %B) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %F = "test.F"(%B) : (tensor<f32>) -> tensor<f32>
  %G = "test.G"(%C) : (tensor<f32>) -> tensor<f32>
  %H = "test.H"(%E) : (tensor<f32>) -> tensor<f32>
  %I = "test.I"(%F, %G) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %J = "test.J"(%H, %I) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %K = "test.K"(%D, %J) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %L = "test.L"(%J, %G) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %M = "test.M"(%K) : (tensor<f32>) -> tensor<f32>
  %N = "test.N"(%L) : (tensor<f32>) -> tensor<f32>
  %O = "test.O"(%M, %N) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %O : tensor<f32>
}
