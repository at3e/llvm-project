#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2) -> ()>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @matmul(%arg0: memref<2x1x4x5xf32>, %arg1: memref<1x3x5x6xf32>) -> memref<2x3x4x6xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3x4x5xf32>
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2], [3]] : memref<2x1x4x5xf32> into memref<2x4x5xf32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapse_shape : memref<2x4x5xf32>) outs(%alloc : memref<2x3x4x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x3x5x6xf32>
    %collapse_shape_1 = memref.collapse_shape %arg1 [[0, 1], [2], [3]] : memref<1x3x5x6xf32> into memref<3x5x6xf32>
    linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapse_shape_1 : memref<3x5x6xf32>) outs(%alloc_0 : memref<2x3x5x6xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %collapse_shape_2 = memref.collapse_shape %alloc [[0, 1], [2], [3]] : memref<2x3x4x5xf32> into memref<6x4x5xf32>
    %collapse_shape_3 = memref.collapse_shape %alloc_0 [[0, 1], [2], [3]] : memref<2x3x5x6xf32> into memref<6x5x6xf32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<6x4x6xf32>
    linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : f32) outs(%alloc_4 : memref<6x4x6xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapse_shape_2, %collapse_shape_3 : memref<6x4x5xf32>, memref<6x5x6xf32>) outs(%alloc_4 : memref<6x4x6xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %0 = arith.mulf %in, %in_5 : f32
      %1 = arith.addf %out, %0 : f32
      linalg.yield %1 : f32
    }
    %expand_shape = memref.expand_shape %alloc_4 [[0, 1], [2], [3]] output_shape [2, 3, 4, 6] : memref<6x4x6xf32> into memref<2x3x4x6xf32>
    return %expand_shape : memref<2x3x4x6xf32>
  }
}

