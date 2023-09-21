// RUN: rm -rf %t && mkdir %t
// RUN: cp %s %t
// RUN: %mlir-bisect --start %t/%basename_t -o /dev/null
// RUN: %mlir-bisect --good %t/%basename_t -o /dev/null
// RUN: %mlir-bisect --bad %t/%basename_t -o /dev/null
// RUN: %mlir-bisect --bad %t/%basename_t -o - | %FileCheck %s

// CHECK: func.func @simple([[ARG0:%[a-z0-9]+]]: tensor<f32>) -> tensor<f32> {
// CHECK-NEXT: [[R1:%[0-9]+]] = "test.B"([[ARG0]]) : (tensor<f32>) -> tensor<f32>
// CHECK-NEXT: [[R3:%[0-9]+]] = "test.D"([[R1:%[0-9]+]]) : (tensor<f32>) -> tensor<f32>
// CHECK-NEXT: return [[R3]] : tensor<f32>
// CHECK-NEXT: }

func.func @simple(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "test.A"(%arg0) : (tensor<f32>) -> tensor<f32>
  %1 = "test.B"(%arg0) : (tensor<f32>) -> tensor<f32>
  %2 = "test.C"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %3 = "test.D"(%1) : (tensor<f32>) -> tensor<f32>
  %4 = "test.E"(%3) : (tensor<f32>) -> tensor<f32>
  %5 = "test.F"(%2, %4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %5 : tensor<f32>
}
