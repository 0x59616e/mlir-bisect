// RUN: rm -rf %t && mkdir %t
// RUN: cp %s %t
// RUN: %mlir-bisect --start %t/%basename_t -o /dev/null
// RUN: %mlir-bisect --bad %t/%basename_t -o /dev/null
// RUN: %mlir-bisect --bad %t/%basename_t -o /dev/null
// RUN: %mlir-bisect --bad %t/%basename_t -o /dev/null
// RUN: cat %t/%basename_t.bisect | %FileCheck %s

// CHECK: func.func @simple([[ARG0:%[a-z0-9]+]]: tensor<f32>) -> tensor<f32> {
// CHECK-NEXT: [[R0:%[0-9]+]] = "test.A"([[ARG0]]) : (tensor<f32>) -> tensor<f32>
// CHECK-NEXT: [[R1:%[0-9]+]] = "test.B"([[ARG0]]) {status = "culprit", which = 0 : i64} : (tensor<f32>) -> tensor<f32>
// CHECK-NEXT: [[R2:%[0-9]+]] = "test.C"([[R0:%[0-9]+]], [[R1]]) {status = "failed", which = 0 : i64} : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT: [[R3:%[0-9]+]] = "test.D"([[R1:%[0-9]+]]) {status = "failed", which = 0 : i64} : (tensor<f32>) -> tensor<f32>
// CHECK-NEXT: [[R4:%[0-9]+]] = "test.E"([[R3:%[0-9]+]]) {status = "failed", which = 0 : i64} : (tensor<f32>) -> tensor<f32>
// CHECK-NEXT: [[R5:%[0-9]+]] = "test.F"([[R2:%[0-9]+]], [[R4]]) {status = "failed", which = 0 : i64} : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT: return %5 : tensor<f32>
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
