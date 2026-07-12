# MLIR pass to flatten function arguments

In this fork, I have introduced a dedicated pass to flatten the memrefs of function arguments, addressing the limitations of the existing [flatten‑memref pass](https://discourse.llvm.org/t/about-flattenmemrefspass/89770). This enhancement aims to ensure complete flattening where required for specific lowering pipelines.

## Example
Here is an example IR of a `matmul` function

```mlir
map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
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
```

The `flatten-memref` pass does not alter the function arguments, but complete flattening can be achieved by using it with `flatten-funcargs` pass.
```diff
@@ -7,38 +7,41 @@
 #map6 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
 #map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
 module {
-  func.func @matmul(%arg0: memref<2x1x4x5xf32>, %arg1: memref<1x3x5x6xf32>) -> memref<2x3x4x6xf32> {
+  func.func @matmul(%arg0: memref<40xf32>, %arg1: memref<90xf32>) -> memref<144xf32> {
     %cst = arith.constant 0.000000e+00 : f32 
+    %expand_shape = memref.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [2, 1, 4, 5] : memref<40xf32> into memref<2x1x4x5xf32>
+    %expand_shape_0 = memref.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 3, 5, 6] : memref<90xf32> into memref<1x3x5x6xf32>
     %alloc = memref.alloc() {alignment = 64 : i64} : memref<120xf32, strided<[1]>>
     %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [0], sizes: [2, 3, 4, 5], strides: [60, 20, 5, 1] : memref<120xf32, strided<[1]>> to memref<2x3x4x5xf32>
-    %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2], [3]] : memref<2x1x4x5xf32> into memref<2x4x5xf32>
+    %collapse_shape = memref.collapse_shape %expand_shape [[0, 1], [2], [3]] : memref<2x1x4x5xf32> into memref<2x4x5xf32>
     linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapse_shape : memref<2x4x5xf32>) outs(%reinterpret_cast : memref<2x3x4x5xf32>) {
     ^bb0(%in: f32, %out: f32):
       linalg.yield %in : f32 
     }   
-    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<180xf32, strided<[1]>>
-    %reinterpret_cast_1 = memref.reinterpret_cast %alloc_0 to offset: [0], sizes: [2, 3, 5, 6], strides: [90, 30, 6, 1] : memref<180xf32, strided<[1]>> to memref<2x3x5x6xf32>
-    %collapse_shape_2 = memref.collapse_shape %arg1 [[0, 1], [2], [3]] : memref<1x3x5x6xf32> into memref<3x5x6xf32>
-    linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapse_shape_2 : memref<3x5x6xf32>) outs(%reinterpret_cast_1 : memref<2x3x5x6xf32>) {
+    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<180xf32, strided<[1]>>
+    %reinterpret_cast_2 = memref.reinterpret_cast %alloc_1 to offset: [0], sizes: [2, 3, 5, 6], strides: [90, 30, 6, 1] : memref<180xf32, strided<[1]>> to memref<2x3x5x6xf32>
+    %collapse_shape_3 = memref.collapse_shape %expand_shape_0 [[0, 1], [2], [3]] : memref<1x3x5x6xf32> into memref<3x5x6xf32>
+    linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapse_shape_3 : memref<3x5x6xf32>) outs(%reinterpret_cast_2 : memref<2x3x5x6xf32>) {
     ^bb0(%in: f32, %out: f32):
       linalg.yield %in : f32 
     }   
-    %collapse_shape_3 = memref.collapse_shape %reinterpret_cast [[0, 1], [2], [3]] : memref<2x3x4x5xf32> into memref<6x4x5xf32>
-    %collapse_shape_4 = memref.collapse_shape %reinterpret_cast_1 [[0, 1], [2], [3]] : memref<2x3x5x6xf32> into memref<6x5x6xf32>
-    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<144xf32, strided<[1]>>
-    %reinterpret_cast_6 = memref.reinterpret_cast %alloc_5 to offset: [0], sizes: [6, 4, 6], strides: [24, 6, 1] : memref<144xf32, strided<[1]>> to memref<6x4x6xf32>
-    linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : f32) outs(%reinterpret_cast_6 : memref<6x4x6xf32>) {
+    %collapse_shape_4 = memref.collapse_shape %reinterpret_cast [[0, 1], [2], [3]] : memref<2x3x4x5xf32> into memref<6x4x5xf32>
+    %collapse_shape_5 = memref.collapse_shape %reinterpret_cast_2 [[0, 1], [2], [3]] : memref<2x3x5x6xf32> into memref<6x5x6xf32>
+    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<144xf32, strided<[1]>>
+    %reinterpret_cast_7 = memref.reinterpret_cast %alloc_6 to offset: [0], sizes: [6, 4, 6], strides: [24, 6, 1] : memref<144xf32, strided<[1]>> to memref<6x4x6xf32>
+    linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : f32) outs(%reinterpret_cast_7 : memref<6x4x6xf32>) {
     ^bb0(%in: f32, %out: f32):
       linalg.yield %in : f32 
     }   
-    linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapse_shape_3, %collapse_shape_4 : memref<6x4x5xf32>, memref<6x5x6xf32>) outs(%reinterpret_cast_6 : memref<6x4x6xf32>) {
-    ^bb0(%in: f32, %in_7: f32, %out: f32):
-      %0 = arith.mulf %in, %in_7 : f32
     }   
-    linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapse_shape_3, %collapse_shape_4 : memref<6x4x5xf32>, memref<6x5x6xf32>) outs(%reinterpret_cast_6 : memref<6x4x6xf32>) {
-    ^bb0(%in: f32, %in_7: f32, %out: f32):
-      %0 = arith.mulf %in, %in_7 : f32
+    linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapse_shape_4, %collapse_shape_5 : memref<6x4x5xf32>, memref<6x5x6xf32>) outs(%reinterpret_cast_7 : memref<6x4x6xf32>) {
+    ^bb0(%in: f32, %in_10: f32, %out: f32):
+      %0 = arith.mulf %in, %in_10 : f32
       %1 = arith.addf %out, %0 : f32
       linalg.yield %1 : f32
     }
-    %expand_shape = memref.expand_shape %reinterpret_cast_6 [[0, 1], [2], [3]] output_shape [2, 3, 4, 6] : memref<6x4x6xf32> into memref<2x3x4x6xf32>
-    return %expand_shape : memref<2x3x4x6xf32>
+    %expand_shape_8 = memref.expand_shape %reinterpret_cast_7 [[0, 1], [2], [3]] output_shape [2, 3, 4, 6] : memref<6x4x6xf32> into memref<2x3x4x6xf32>
+    %collapse_shape_9 = memref.collapse_shape %expand_shape_8 [[0, 1, 2, 3]] : memref<2x3x4x6xf32> into memref<144xf32>
+    return %collapse_shape_9 : memref<144xf32>
   }
 }
```
