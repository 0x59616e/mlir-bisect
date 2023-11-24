// RUN: rm -rf %t && mkdir %t
// RUN: cp %s %t
// RUN: %mlir-bisect --start %t/%basename_t -o /dev/null
// RUN: %mlir-bisect --good %t/%basename_t -o /dev/null
// RUN: %mlir-bisect --good %t/%basename_t -o /dev/null
// RUN: %mlir-bisect --bad %t/%basename_t -o /dev/null
// RUN: %mlir-bisect --good %t/%basename_t -o - | FileCheck %s


// CHECK: func.func @hard([[arg:%.+]]: tensor<f32>) -> tensor<f32> {
// CHECK-NEXT:   [[r0:%.+]]:2 = "test.A"([[arg]]) : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
// CHECK-NEXT:   [[r1:%.+]] = "test.C"([[r0]]#1) : (tensor<f32>) -> tensor<f32>
// CHECK-NEXT:   [[r2:%.+]] = "test.H"([[r1]]) : (tensor<f32>) -> tensor<f32>
// CHECK-NEXT:   return [[r2]] : tensor<f32>
// CHECK-NEXT: }

func.func @hard(%arg0: tensor<f32>) -> tensor<f32> {
  %0:2 = "test.A"(%arg0) : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  %1:3 = "test.B"(%0#0) : (tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>)
  %2   = "test.C"(%0#1) : (tensor<f32>) -> (tensor<f32>)
  %3   = "test.D"(%1#0, %1#1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %4   = "test.E"(%1#1, %1#2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %5   = "test.F"(%1#2) : (tensor<f32>) -> tensor<f32>
  %6   = "test.G"(%3, %4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %7   = "test.H"(%2) : (tensor<f32>) -> tensor<f32>
  %8   = "test.I"(%6, %7) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
  return %8 : tensor<f32>
}
