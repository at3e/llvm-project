# MLIR pass to flatten function arguments

In this fork, I have introduced a dedicated pass to flatten the memrefs of function arguments, addressing the limitations of the existing [flatten‑memref pass](https://discourse.llvm.org/t/about-flattenmemrefspass/89770). This enhancement aims to ensure complete flattening where required for specific lowering pipelines.

## Example



