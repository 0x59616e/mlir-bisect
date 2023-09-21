// RUN: %mlir-bisect --dump-idom %s | %FileCheck %s
// CHECK-DAG: [[D:%[0-9]+]] = "test.D"
// CHECK-DAG: [[B:%[0-9]+]] = "test.B"
// CHECK-DAG: [[A:%[0-9]+]] = "test.A"
// CHECK-DAG: [[C:%[0-9]+]] = "test.C"
// CHECK-DAG: [[E:%[0-9]+]] = "test.E"
// CHECK-DAG: [[ARG0:<block argument> of type 'tensor<f32>' at index: 0]]
// CHECK-DAG: [[F:%[0-9]+]] = "test.F"
// CHECK: ---
// CHECK-DAG: <<NULL VALUE>> => [[ARG0]]
// CHECK-DAG: [[ARG0]] => [[C]]
// CHECK-DAG: [[ARG0]] => [[B]]
// CHECK-DAG: [[ARG0]] => [[F]]
// CHECK-DAG: [[ARG0]] => [[A]]
// CHECK-DAG: [[B]]{{.+}}=> [[D]]
// CHECK-DAG: [[D]]{{.+}} => [[E]]

func.func @identity(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "test.A"(%arg0) : (tensor<f32>) -> tensor<f32>
  %1 = "test.B"(%arg0) : (tensor<f32>) -> tensor<f32>
  %2 = "test.C"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %3 = "test.D"(%1) : (tensor<f32>) -> tensor<f32>
  %4 = "test.E"(%3) : (tensor<f32>) -> tensor<f32>
  %5 = "test.F"(%2, %4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %5 : tensor<f32>
}
