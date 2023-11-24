// RUN: rm -rf %t && mkdir %t
// RUN: cp %s %t
// RUN: %mlir-bisect --start %t/%basename_t -o /dev/null
// RUN: %mlir-bisect --good %t/%basename_t -o /dev/null
// RUN: %mlir-bisect --bad %t/%basename_t -o - | %FileCheck %s

// CHECK: func.func @simple([[ARG0:%.+]]: tensor<f32>) -> tensor<f32> {
// CHECK-NEXT: [[R0:%.+]]:2 = "test.A"([[ARG0]]) : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
// CHECK-NEXT: return [[R0]]#1 : tensor<f32>
// CHECK-NEXT: }

func.func @simple(%arg0: tensor<f32>) -> tensor<f32> {
  %0, %1 = "test.A"(%arg0) : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  %2 = "test.B"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %2 : tensor<f32>
}
