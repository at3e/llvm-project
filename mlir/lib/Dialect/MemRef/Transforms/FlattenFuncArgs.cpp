#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_FLATTENFUNCARGSPASS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;

namespace {

/// Returns failure if `type` is not a statically shaped,
/// contiguous row-major memref (e.g. it has a non-default dimension
/// order or padding), since such a memref cannot be losslessly reshaped to
/// 1-D via a plain reassociation.
FailureOr<MemRefType> getFlattenedType(MemRefType type, int64_t totalSize) {
  if (!type.hasStaticShape())
    return failure();

  int64_t offset;
  SmallVector<int64_t> strides;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return failure();

  // Allow flattening if the memref is provably contiguous row-major, or if
  // its strides are entirely dynamic.
  bool allDynamicStrides = llvm::all_of(strides, ShapedType::isDynamic);
  if (!allDynamicStrides && !memref::isStaticShapeAndContiguousRowMajor(type))
    return failure();

  auto layout = StridedLayoutAttr::get(type.getContext(), offset, {1});
  return MemRefType::get({totalSize}, type.getElementType(), layout,
                          type.getMemorySpace());
}

void rewriteArgs(func::FuncOp funcOp) {
  if (funcOp.empty())
      return;

  Block &entryBlock = funcOp.front();

  SmallVector<Type> rewriteArgTypes;
  bool is_flat = false;

  OpBuilder builder(&entryBlock, entryBlock.begin());

  for (BlockArgument arg : entryBlock.getArguments()) {
    auto memrefType = dyn_cast<MemRefType>(arg.getType());
    // Check if the arg is not memref at all. If it is a memref, but already
    // rank-0 or rank-1, there is no need to flatten.
    if (!memrefType || memrefType.getRank() <= 1) {
      rewriteArgTypes.push_back(arg.getType());
      continue;
    }

    int64_t totalSize = 1;
    for (int64_t dim : memrefType.getShape())
      totalSize *= dim;

    // If memref has non-static shape and non-contiguous strides, then rewriteArgs
    // does not touch the offset and preserves the original arg type.
    FailureOr<MemRefType> flatType = getFlattenedType(memrefType, totalSize);
    if (failed(flatType)) {
      rewriteArgTypes.push_back(arg.getType());
      continue;
    }

    // Retype the block argument to the flat 1-D memref.
    arg.setType(*flatType);
    rewriteArgTypes.push_back(*flatType);

    is_flat = true;

    // Re-expand the 1-D arg back to the original N-D shape so that every
    // use in the function body continues to see a correctly-typed value.
    // TO DO : memrefs across function body should be uniformly flattened.
    SmallVector<ReassociationIndices> reassociationDims;
    ReassociationIndices tempDims;

    for (int64_t i = 0; i < memrefType.getRank(); ++i)
      tempDims.push_back(i);
    
    reassociationDims.push_back(tempDims);

    // Compute the result type from the flat source itself, rather than
    // reusing the original `memrefType`: when the source's strides were
    // originally fully dynamic (see getFlattenedType), re-expanding a
    // provably-contiguous flat memref yields canonical strides. 
    FailureOr<MemRefType> expandedType =
        memref::ExpandShapeOp::computeExpandedType(
            *flatType, memrefType.getShape(), reassociationDims);
    assert(succeeded(expandedType) &&
           "flattening succeeded, so re-expanding to the same shape must too");

    auto expandOp = memref::ExpandShapeOp::create(
        builder, funcOp.getLoc(), *expandedType, arg, reassociationDims);

      // Replace every use of arg with the N-D view, except inside expandOp
      // itself (where arg is the operand).
      arg.replaceAllUsesExcept(expandOp.getResult(), expandOp);
  }

  if (is_flat) {
   funcOp.setFunctionType(FunctionType::get(
        funcOp.getContext(), rewriteArgTypes, funcOp.getFunctionType().getResults()));
  }

  // Flatten the return type argument

  // Type resultType = funcOp.getFunctionType().getResult(0);
  SmallVector<Type> rewriteResTypes;
  is_flat = false;

  for (Type arg : funcOp.getFunctionType().getResults()) {

    auto resMemrefType = dyn_cast<MemRefType>(arg);

    if (!resMemrefType || resMemrefType.getRank() <= 1) {
      rewriteResTypes.push_back(arg);
      continue;
    }

    int64_t totalSize = 1;
    for (int64_t dim : resMemrefType.getShape())
      totalSize *= dim;

    FailureOr<MemRefType> flatResultType =
        getFlattenedType(resMemrefType, totalSize);
    if (failed(flatResultType)) {
      rewriteResTypes.push_back(arg);
      continue;
    }

    rewriteResTypes.push_back(*flatResultType);
    is_flat = true;
  }

  if (is_flat) {
    funcOp.walk([&](func::ReturnOp returnOp) {
      OpBuilder retBuilder(returnOp);
      for (unsigned idx = 0; idx < returnOp.getNumOperands(); ++idx) {
        Value operand = returnOp.getOperand(idx);
        auto flatResType = dyn_cast<MemRefType>(rewriteResTypes[idx]);
        auto origResType = dyn_cast<MemRefType>(operand.getType());
        if (!flatResType || !origResType || origResType == flatResType)
          continue;

        // Reassociation: all N source dims collapse into the single
        // flat output dim 0.
        SmallVector<ReassociationIndices> reassociationDims;
        ReassociationIndices tmpDims;
        for (int64_t i = 0; i < origResType.getRank(); ++i)
          tmpDims.push_back(i);
        reassociationDims.push_back(tmpDims);

        auto collapseOp = memref::CollapseShapeOp::create(
            retBuilder, returnOp.getLoc(), flatResType, operand,
            reassociationDims);
        returnOp.setOperand(idx, collapseOp.getResult());
      }
    });

    funcOp.setFunctionType(FunctionType::get(
        funcOp.getContext(), rewriteArgTypes, rewriteResTypes));
  }

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
