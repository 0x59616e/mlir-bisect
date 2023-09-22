// RUN: rm -rf %t && mkdir %t
// RUN: cp %s %t
// RUN: %mlir-bisect --start %t/%basename_t -o /dev/null
// RUN: %mlir-bisect --bad %t/%basename_t -o /dev/null
// RUN: %mlir-bisect --bad %t/%basename_t -o /dev/null
// RUN: %mlir-bisect --bad %t/%basename_t -o - | %FileCheck %s

// CHECK: func.func @hard([[ARG0:%[a-z0-9]+]]: tensor<f32>) -> tensor<f32> {
// CHECK-NEXT:   [[A:%[0-9]+]] = "test.A"([[ARG0]]) : (tensor<f32>) -> tensor<f32>
// CHECK-NEXT:   return [[A]] : tensor<f32>
// CHECK-NEXT: }


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
