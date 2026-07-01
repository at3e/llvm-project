#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_FLATTENFUNCARGSPASS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;

namespace {

void rewriteArgs(func::FuncOp funcOp) {
  if (funcOp.empty())
      return;

  Block &entryBlock = funcOp.front();

  SmallVector<Type> rewriteArgTypes;
  bool is_flat = false;

  OpBuilder builder(&entryBlock, entryBlock.begin());

  for (BlockArgument arg : entryBlock.getArguments()) {
    auto memrefType = dyn_cast<MemRefType>(arg.getType());
    if (!memrefType || memrefType.getRank() <= 1 || !memrefType.hasStaticShape()) {
      rewriteArgTypes.push_back(arg.getType());
      continue;
    } 

    int64_t totalSize = 1;
    for (int64_t dim : memrefType.getShape())
      totalSize *= dim;

    auto flatType = MemRefType::get({totalSize}, memrefType.getElementType(),
                                     /*layout=*/MemRefLayoutAttrInterface{},
                                     memrefType.getMemorySpace());
    // Retype the block argument to the flat 1-D memref.
    arg.setType(flatType);
    rewriteArgTypes.push_back(flatType);

    is_flat = true;

    // Re-expand the 1-D arg back to the original N-D shape so that every
    // use in the function body continues to see a correctly-typed value.
    // Reassociation: all N output dims come from the single input dim 0.
    SmallVector<ReassociationIndices> reassoc;
    ReassociationIndices allDims;

    for (int64_t i = 0; i < memrefType.getRank(); ++i)
      allDims.push_back(i);
    
    reassoc.push_back(allDims);

    auto expandOp = builder.create<memref::ExpandShapeOp>(
        funcOp.getLoc(), memrefType, arg, reassoc);

      // Replace every use of arg with the N-D view, except inside expandOp
      // itself (where arg is the operand).
      arg.replaceAllUsesExcept(expandOp.getResult(), expandOp);
  }

  if (is_flat) {
    funcOp.setFunctionType(FunctionType::get(
        funcOp.getContext(), rewriteArgTypes, funcOp.getResultTypes()));
  }

  return;

}

struct FlattenFuncArgs
    : public mlir::memref::impl::FlattenFuncArgsPassBase<FlattenFuncArgs> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    rewriteArgs(funcOp);
  }
};

} // namespace

using namespace mlir;
