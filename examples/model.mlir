#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
module {
  func.func @matmul(%arg0: tensor<2x1x4x5xf32>, %arg1: tensor<1x3x5x6xf32>) -> tensor<2x3x4x6xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x3x4x5xf32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2], [3]] : tensor<2x1x4x5xf32> into tensor<2x4x5xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed : tensor<2x4x5xf32>) outs(%0 : tensor<2x3x4x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<2x3x4x5xf32>
    %2 = tensor.empty() : tensor<2x3x5x6xf32>
    %collapsed_0 = tensor.collapse_shape %arg1 [[0, 1], [2], [3]] : tensor<1x3x5x6xf32> into tensor<3x5x6xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_0 : tensor<3x5x6xf32>) outs(%2 : tensor<2x3x5x6xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<2x3x5x6xf32>
    %collapsed_1 = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<2x3x4x5xf32> into tensor<6x4x5xf32>
    %collapsed_2 = tensor.collapse_shape %3 [[0, 1], [2], [3]] : tensor<2x3x5x6xf32> into tensor<6x5x6xf32>
    %4 = tensor.empty() : tensor<6x4x6xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<6x4x6xf32>) -> tensor<6x4x6xf32>
    %6 = linalg.batch_matmul ins(%collapsed_1, %collapsed_2 : tensor<6x4x5xf32>, tensor<6x5x6xf32>) outs(%5 : tensor<6x4x6xf32>) -> tensor<6x4x6xf32>
    %expanded = tensor.expand_shape %6 [[0, 1], [2], [3]] output_shape [2, 3, 4, 6] : tensor<6x4x6xf32> into tensor<2x3x4x6xf32>
    return %expanded : tensor<2x3x4x6xf32>
  }
}
