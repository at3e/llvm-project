# `flatten-contiguous-memref`

`flatten-contiguous-memref` (`FlattenContiguousMemrefPass`, defined in
`FlattenContiguousMemref.cpp`) rewrites a function so that every statically
shaped, contiguous multi-dimensional memref it owns becomes a plain 1-D memref,
and every access to it uses a single linearized index.

It runs on `func.func` and is meant for consumers that want flat memories with
no reshaping or reinterpreting ops left in the IR, such as CIRCT's
`--lower-cf-to-handshake`.

```
mlir-opt --pass-pipeline='builtin.module(func.func(flatten-contiguous-memref))' in.mlir
```

## What it does

For each `func.func` with a body:

1. **Arguments.** Each multi-dimensional memref argument that can be flattened
   (see [Which memrefs are flattened](#which-memrefs-are-flattened)) is retyped
   in place to its 1-D form, e.g. `memref<2x2xi32>` -> `memref<4xi32>`.
2. **Local buffers.** Each such `memref.alloc` / `memref.alloca` result is
   retyped in place the same way. The `alignment` attribute is removed from
   every `alloc`/`alloca` in the function, because CIRCT's parser rejects it.
3. **Accesses.** Users of a retyped value are rewritten to index the flat
   memref directly with `sum(index[k] * stride[k])`, built from
   `arith.muli`/`arith.addi`. Size-1 dimensions are skipped. This covers:
   - `memref.load`, `memref.store`
   - `vector.load`, `vector.store`, `vector.maskedload`, `vector.maskedstore`
     with a 1-D vector
   - `vector.transfer_read`, `vector.transfer_write` with a 1-D vector, a
     minor-identity permutation map, no mask, and all dims in bounds
4. **Other users.** `func.return`, `memref.dealloc` and calls to functions
   with a body accept the flat value unchanged. Any other user gets an N-D view
   from a single `memref.expand_shape` inserted after the definition.
5. **Calls.** Multi-dimensional memref operands of `func.call` (to a function
   with a body) are collapsed with `memref.collapse_shape` to match the
   callee's flattened parameters. `func.call_indirect` is handled only when
   the callee comes from a `func.constant` naming a known, non-external
   function; that constant's type is updated in step. External declarations
   are left untouched.
6. **Results are dropped.** Every `func.return` becomes an empty return and the
   function type loses its results (and result attributes). Calls to functions
   with a body are rebuilt without results. If any of those call results is
   used, the pass emits an error and fails.

### Which memrefs are flattened

A memref is flattened only if:

- its rank is greater than 1,
- its shape is fully static, and
- its strides are static and prove the data is one contiguous row-major block:
  `stride[N-1] == 1` and `stride[i] == stride[i+1] * shape[i+1]`. The stride of
  a size-1 dimension is ignored.

The flat type has `shape[0] * ... * shape[N-1]` elements. An identity layout
flattens to a 1-D memref with no layout; a strided layout flattens to
`strided<[1], offset: <original offset>>`. The memory space is kept. Anything
else (dynamic shapes, padded or permuted layouts) is left unchanged.

## Example

Input:

```mlir
func.func @matadd(%arg0: memref<2x2xi32>, %arg1: memref<2x2xi32>) -> memref<2x2xi32> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x2xi32>
  ...
      %0 = memref.load %arg0[%i, %j] : memref<2x2xi32>
      %1 = memref.load %arg1[%i, %j] : memref<2x2xi32>
      %2 = arith.addi %0, %1 : i32
      memref.store %2, %alloc[%i, %j] : memref<2x2xi32>
  ...
  return %alloc : memref<2x2xi32>
}
```

After `flatten-contiguous-memref`:

```mlir
func.func @matadd(%arg0: memref<4xi32>, %arg1: memref<4xi32>) {
  %alloc = memref.alloc() : memref<4xi32>
  ...
      %c2 = arith.constant 2 : index
      %r = arith.muli %i, %c2 : index
      %idx = arith.addi %r, %j : index
      %0 = memref.load %arg0[%idx] : memref<4xi32>
      ...
      memref.store %2, %alloc[%idx2] : memref<4xi32>
  ...
  return
}
```

Because results are dropped, a value a function used to return is no longer
visible to its caller. To keep an output, pass its buffer in as an argument
(e.g. `func.func @matadd(%a, %b, %out: memref<2x2xi32>)`).

## Comparison with `flatten-memref`

`flatten-memref` (`FlattenMemrefsPass`, `FlattenMemRefs.cpp`) is the upstream
pass for the same basic job. The two differ mainly in *where* the flat view
comes from: `flatten-memref` keeps every memref's type and builds a flat view
next to each access, while `flatten-contiguous-memref` changes the memref's own
type once and rewrites its accesses to use it.

| | `flatten-memref` | `flatten-contiguous-memref` |
|---|---|---|
| Anchor | Any op (`Pass<"flatten-memref">`) | `func.func` |
| Mechanism | Greedy rewrite patterns, also exposed as `populateFlattenMemrefsPatterns` | One walk per function; no public pattern set |
| Function arguments | Type unchanged | Retyped to 1-D |
| Function results | Unchanged | Dropped |
| `alloc`/`alloca` | Replaced by a 1-D alloc plus a `memref.reinterpret_cast` back to the N-D type | Result retyped to 1-D in place; `alignment` removed |
| Each access | Gets its own `memref.extract_strided_metadata` + `memref.reinterpret_cast` to a 1-D view, plus index arithmetic (`affine.apply`) | Indexes the already-flat memref; index built with `arith.muli`/`arith.addi` |
| Ops left behind | `extract_strided_metadata`, `reinterpret_cast`, `affine.apply` | None for supported accesses; `expand_shape` only for unsupported users |
| Supported accesses | Any op implementing `IndexedAccessOpInterface`, `IndexedMemCopyOpInterface` (e.g. `memref.dma_start`) or `VectorTransferOpInterface` | A fixed list of `memref` and `vector` ops (see above) |
| Dynamic shapes / strides | Supported | Not flattened |
| Non-contiguous strided layouts | Supported (flat size covers the largest `size * stride`) | Not flattened |
| Element types | Integer and float only | Any |
| Calls | Not handled | `func.call` and `func.constant`-based `func.call_indirect` operands collapsed to match flattened callees |

In short:

- Use **`flatten-memref`** for a general, composable rewrite that keeps
  function interfaces intact and works with dynamic or strided memrefs.
- Use **`flatten-contiguous-memref`** when the target wants truly 1-D memories
  end to end, with flat function signatures and no `reinterpret_cast` /
  `extract_strided_metadata` / `affine.apply` in the output, and the inputs
  are statically shaped and contiguous. This is what CIRCT's handshake
  lowering expects.

It also replaces running `flatten-funcargs` followed by `flatten-memref`. That
combination leaves an `expand_shape` at the function entry and a
`reinterpret_cast` per access, while this pass produces neither for supported
accesses.
