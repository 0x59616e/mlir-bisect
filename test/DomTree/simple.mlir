// RUN: %mlir-bisect --dump-idom %s | %FileCheck %s

// CHECK: [[ARG0:<block argument> of type 'tensor<f32>' at index: 0]]
// CHECK-NEXT: [[A:%[0-9]+]] = "test.A"
// CHECK-NEXT: [[B:%[0-9]+]] = "test.B"
// CHECK-NEXT: [[C:%[0-9]+]] = "test.C"
// CHECK-NEXT: [[D:%[0-9]+]] = "test.D"
// CHECK-NEXT: [[E:%[0-9]+]] = "test.E"
// CHECK-NEXT: [[F:%[0-9]+]] = "test.F"
// CHECK-NEXT: ---
// CHECK-NEXT: <<NULL VALUE>> => [[ARG0]]
// CHECK-NEXT: [[ARG0]] => [[A]]
// CHECK-NEXT: [[ARG0]] => [[B]]
// CHECK-NEXT: [[ARG0]] => [[C]]
// CHECK-NEXT: [[B]]{{.+}}=> [[D]]
// CHECK-NEXT: [[D]]{{.+}} => [[E]]
// CHECK-NEXT: [[ARG0]] => [[F]]

func.func @identity(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "test.A"(%arg0) : (tensor<f32>) -> tensor<f32>
  %1 = "test.B"(%arg0) : (tensor<f32>) -> tensor<f32>
  %2 = "test.C"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %3 = "test.D"(%1) : (tensor<f32>) -> tensor<f32>
  %4 = "test.E"(%3) : (tensor<f32>) -> tensor<f32>
  %5 = "test.F"(%2, %4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %5 : tensor<f32>
}
