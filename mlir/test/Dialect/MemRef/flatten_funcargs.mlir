// RUN: mlir-opt --flatten-funcargs %s --split-input-file --verify-diagnostics | FileCheck %s

func.func @load_scalar_from_memref(%input: memref<4x8xf32, strided<[8, 1], offset: 100>>) -> f32 {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %value = memref.load %input[%c1, %c2] : memref<4x8xf32, strided<[8, 1], offset: 100>>
  return %value : f32
}

// CHECK-LABEL: func @load_scalar_from_memref
// CHECK-SAME: memref<32xf32, strided<[1], offset: 100>> to memref<32xf32, strided<[1], offset: 100>>

// -----

func.func @load_scalar_from_memref_dynamic_dim(%input: memref<?x?xf32, strided<[?, ?], offset: ?>>, %row: index, %col: index) -> f32 {
  %value = memref.load %input[%col, %row] : memref<?x?xf32, strided<[?, ?], offset: ?>>
  return %value : f32
}

// CHECK-LABEL: func @load_scalar_from_memref
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32, strided<[?, ?], offset: ?>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)

// -----

func.func @load_scalar_from_memref_static_dim(%input: memref<8x12xf32, strided<[24, 2], offset: 100>>) -> f32 {
   %c7 = arith.constant 7 : index
   %c10 = arith.constant 10 : index
  %value = memref.load %input[%c7, %c10] : memref<8x12xf32, strided<[24, 2], offset: 100>>
  return %value : f32
}

// CHECK-LABEL: func @load_scalar_from_memref_static_dim
// CHECK-SAME: (%[[ARG0:.*]]: memref<8x12xf32, strided<[24, 2], offset: 100>>)

// -----

func.func @store_scalar_from_memref_padded(%input: memref<4x8xf32, strided<[18, 2], offset: 100>>, %row: index, %col: index, %value: f32) {
  memref.store %value, %input[%col, %row] : memref<4x8xf32, strided<[18, 2], offset: 100>>
  return
}
// CHECK: #[[MAP:.*]] = affine_map<()[s0, s1] -> (s0 * 18 + s1 * 2)>
// CHECK: func @store_scalar_from_memref_padded
// CHECK-SAME: (%[[ARG0:.*]]: memref<4x8xf32, strided<[18, 2], offset: 100>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: f32)
