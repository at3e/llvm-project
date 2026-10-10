//===- FlattenContiguousMemref.cpp - Flatten contiguous memrefs to 1-D ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Combines flatten-funcargs and flatten-memref: every statically shaped,
// contiguous multi-dimensional memref that a function owns (its arguments and
// its memref.alloc/alloca results) is retyped to 1-D, and every load/store on
// it is rewritten to index the flat memref directly with a linearized index.
// Unlike running the two passes back to back, no expand_shape or
// reinterpret_cast is produced for these accesses.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_FLATTENCONTIGUOUSMEMREFPASS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;

namespace {

/// Returns true if the N-D memref described by `shape` and `strides` occupies
/// one contiguous row-major block of memory, i.e. walking from the innermost
/// dimension outwards, every stride equals the product of the sizes of all
/// dimensions inside it:
///
///   strides[N-1] == 1
///   strides[i]   == strides[i+1] * shape[i+1]
///
/// The stride of a size-1 dimension is never used to address memory, so it is
/// ignored wherever that dimension appears. Dynamic sizes or strides cannot be
/// proven contiguous and are rejected.
bool hasContiguousStrides(ArrayRef<int64_t> shape,
                          ArrayRef<int64_t> strides) {
  assert(shape.size() == strides.size() && "expected one stride per dim");

  int64_t expectedStride = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    if (ShapedType::isDynamic(shape[i]))
      return false;
    if (shape[i] == 1)
      continue;
    if (ShapedType::isDynamic(strides[i]) || strides[i] != expectedStride)
      return false;
    expectedStride *= shape[i];
  }
  return true;
}

/// Returns the 1-D type `type` flattens to, or failure if `type` is not a
/// statically shaped memref whose strides prove it contiguous (e.g. it has a
/// non-default dimension order or padding), since such a memref cannot be
/// losslessly reshaped to 1-D.
FailureOr<MemRefType> getFlattenedType(MemRefType type) {
  if (!type.hasStaticShape())
    return failure();

  int64_t offset;
  SmallVector<int64_t> strides;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return failure();

  // Only flatten memrefs whose strides prove the data is contiguous.
  if (!hasContiguousStrides(type.getShape(), strides))
    return failure();

  // Match what collapse_shape/expand_shape's own verifiers infer: an
  // identity-layout source collapses to a plain (no layout attr) 1-D type,
  // not an explicit `strided<[1], offset: 0>` -- the two are structurally
  // distinct MemRefTypes even though they're semantically identical, and the
  // verifier rejects the latter as a mismatch against its own inferred type.
  MemRefLayoutAttrInterface layout;
  if (!type.getLayout().isIdentity())
    layout = StridedLayoutAttr::get(type.getContext(), offset, {1});

  return MemRefType::get({type.getNumElements()}, type.getElementType(),
                         layout, type.getMemorySpace());
}

/// Returns the reassociation that maps all `rank` dims onto a single dim.
SmallVector<ReassociationIndices> getFullReassociation(int64_t rank) {
  ReassociationIndices dims;
  for (int64_t i = 0; i < rank; ++i)
    dims.push_back(i);
  return {dims};
}

/// Builds the index into the flat form of `type` that addresses the element
/// at `indices`: sum(indices[k] * strides[k]). Size-1 dims are skipped, since
/// an in-bounds index into one is always 0.
Value linearizeIndex(OpBuilder &builder, Location loc, MemRefType type,
                     ValueRange indices) {
  int64_t offset;
  SmallVector<int64_t> strides;
  LogicalResult stridesOk = type.getStridesAndOffset(strides, offset);
  assert(succeeded(stridesOk) && "flattened types have static strides");
  (void)stridesOk;

  Value linear;
  for (auto [dim, stride, index] :
       llvm::zip_equal(type.getShape(), strides, indices)) {
    if (dim == 1)
      continue;
    Value term = index;
    if (stride != 1) {
      Value strideValue =
          arith::ConstantIndexOp::create(builder, loc, stride).getResult();
      term = arith::MulIOp::create(builder, loc, index, strideValue);
    }
    linear = linear ? arith::AddIOp::create(builder, loc, linear, term)
                    : term;
  }
  if (!linear)
    linear = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  return linear;
}

/// Only a 1-D vector covers contiguous elements in both the N-D memref and
/// its flat form, so vector accesses of any other shape are not rewritten.
bool isFlattenableVector(VectorType type) { return type.getRank() == 1; }

/// Rewrites `op`, an access whose base memref `flat` was just retyped from
/// `origType` to its 1-D form, to index `flat` directly. Returns failure,
/// leaving `op` untouched, if `op` is not an access that can be rewritten.
LogicalResult rewriteAccess(Operation *op, Value flat, MemRefType origType) {
  OpBuilder builder(op);
  Location loc = op->getLoc();
  auto linearize = [&](ValueRange indices) {
    return linearizeIndex(builder, loc, origType, indices);
  };

  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](memref::LoadOp oper) {
        if (oper.getMemref() != flat)
          return failure();
        auto newLoad = memref::LoadOp::create(
            builder, loc, oper->getResultTypes(), flat,
            ValueRange{linearize(oper.getIndices())});
        newLoad->setAttrs(oper->getAttrs());
        oper.getResult().replaceAllUsesWith(newLoad.getResult());
        oper.erase();
        return success();
      })
      .Case([&](memref::StoreOp oper) {
        if (oper.getMemref() != flat)
          return failure();
        auto newStore = memref::StoreOp::create(
            builder, loc, oper.getValueToStore(), flat,
            ValueRange{linearize(oper.getIndices())});
        newStore->setAttrs(oper->getAttrs());
        oper.erase();
        return success();
      })
      .Case([&](vector::LoadOp oper) {
        if (oper.getBase() != flat ||
            !isFlattenableVector(oper.getVectorType()))
          return failure();
        auto newLoad = vector::LoadOp::create(
            builder, loc, oper->getResultTypes(), flat,
            ValueRange{linearize(oper.getIndices())});
        newLoad->setAttrs(oper->getAttrs());
        oper.getResult().replaceAllUsesWith(newLoad.getResult());
        oper.erase();
        return success();
      })
      .Case([&](vector::StoreOp oper) {
        if (oper.getBase() != flat ||
            !isFlattenableVector(oper.getVectorType()))
          return failure();
        auto newStore = vector::StoreOp::create(
            builder, loc, oper.getValueToStore(), flat,
            ValueRange{linearize(oper.getIndices())});
        newStore->setAttrs(oper->getAttrs());
        oper.erase();
        return success();
      })
      .Case([&](vector::MaskedLoadOp oper) {
        if (oper.getBase() != flat ||
            !isFlattenableVector(oper.getVectorType()))
          return failure();
        auto newLoad = vector::MaskedLoadOp::create(
            builder, loc, oper.getType(), flat,
            ValueRange{linearize(oper.getIndices())}, oper.getMask(),
            oper.getPassThru());
        newLoad->setAttrs(oper->getAttrs());
        oper.getResult().replaceAllUsesWith(newLoad.getResult());
        oper.erase();
        return success();
      })
      .Case([&](vector::MaskedStoreOp oper) {
        if (oper.getBase() != flat ||
            !isFlattenableVector(oper.getVectorType()))
          return failure();
        auto newStore = vector::MaskedStoreOp::create(
            builder, loc, flat, ValueRange{linearize(oper.getIndices())},
            oper.getMask(), oper.getValueToStore());
        newStore->setAttrs(oper->getAttrs());
        oper.erase();
        return success();
      })
      .Case<vector::TransferReadOp, vector::TransferWriteOp>(
          [&](auto oper) -> LogicalResult {
            // The flat access must read/write the same elements, so require
            // a 1-D vector along the innermost dim (minor identity map), no
            // mask, and all accesses in bounds.
            if (oper.getBase() != flat ||
                !isFlattenableVector(oper.getVectorType()) ||
                oper.getMask() ||
                !oper.getPermutationMap().isMinorIdentity() ||
                llvm::any_of(oper.getInBounds(), [](Attribute attr) {
                  return !cast<BoolAttr>(attr).getValue();
                }))
              return failure();

            Value index = linearize(oper.getIndices());
            if constexpr (std::is_same_v<decltype(oper),
                                         vector::TransferReadOp>) {
              auto newRead = vector::TransferReadOp::create(
                  builder, loc, oper.getVectorType(), flat, ValueRange{index},
                  oper.getPadding());
              newRead.setInBoundsAttr(oper.getInBoundsAttr());
              oper.getResult().replaceAllUsesWith(newRead.getResult());
            } else {
              auto newWrite = vector::TransferWriteOp::create(
                  builder, loc, oper.getVector(), flat, ValueRange{index});
              newWrite.setInBoundsAttr(oper.getInBoundsAttr());
            }
            oper.erase();
            return success();
          })
      .Default([](Operation *) { return failure(); });
}

/// Returns true if `callOp` calls a function with a body in `moduleOp`. This
/// same pass flattens that callee's parameters, so its call sites can pass
/// flat memrefs.
bool callsFlattenedFunc(func::CallOp callOp, ModuleOp moduleOp) {
  auto callee =
      moduleOp ? moduleOp.lookupSymbol<func::FuncOp>(callOp.getCalleeAttr())
               : nullptr;
  return callee && !callee.isExternal();
}

/// Fixes up the users of `flat`, a value that was just retyped from the N-D
/// `origType` to its 1-D form:
///  - loads/stores are rewritten to index `flat` directly,
///  - calls to flattened functions, returns (which this pass empties) and
///    deallocs accept the flat value as is,
///  - any other user gets an N-D view of `flat` from a memref.expand_shape.
void rewriteUsers(Value flat, MemRefType origType, ModuleOp moduleOp) {
  llvm::SetVector<Operation *> users(flat.user_begin(), flat.user_end());
  Value expanded;
  for (Operation *user : users) {
    if (succeeded(rewriteAccess(user, flat, origType)))
      continue;
    if (isa<func::ReturnOp, memref::DeallocOp>(user))
      continue;
    if (auto callOp = dyn_cast<func::CallOp>(user);
        callOp && callsFlattenedFunc(callOp, moduleOp))
      continue;

    if (!expanded) {
      auto flatType = cast<MemRefType>(flat.getType());
      SmallVector<ReassociationIndices> reassociation =
          getFullReassociation(origType.getRank());
      // Compute the result type from the flat source itself, rather than
      // reusing `origType`: size-1 dims may carry arbitrary strides (see
      // hasContiguousStrides), while re-expanding a contiguous flat memref
      // yields canonical strides.
      FailureOr<MemRefType> expandedType =
          memref::ExpandShapeOp::computeExpandedType(
              flatType, origType.getShape(), reassociation);
      assert(succeeded(expandedType) &&
             "flattening succeeded, so re-expanding to the same shape must "
             "too");
      OpBuilder builder(flat.getContext());
      builder.setInsertionPointAfterValue(flat);
      expanded = memref::ExpandShapeOp::create(builder, flat.getLoc(),
                                               *expandedType, flat,
                                               reassociation);
    }
    user->replaceUsesOfWith(flat, expanded);
  }
}

LogicalResult flattenContiguousMemref(func::FuncOp funcOp) {
  if (funcOp.empty())
    return success();

  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  Block &entryBlock = funcOp.front();

  // Flatten the arguments: retype each flattenable multi-dim memref argument
  // to 1-D in place, then fix up its users.
  SmallVector<Type> rewriteArgTypes;
  for (BlockArgument arg : entryBlock.getArguments()) {
    auto memrefType = dyn_cast<MemRefType>(arg.getType());
    // Non-memrefs and memrefs that are already rank-0 or rank-1 need no
    // flattening.
    if (!memrefType || memrefType.getRank() <= 1) {
      rewriteArgTypes.push_back(arg.getType());
      continue;
    }

    // If the memref has a non-static shape or its strides are not provably
    // contiguous (see hasContiguousStrides), keep the original arg type.
    FailureOr<MemRefType> flatType = getFlattenedType(memrefType);
    if (failed(flatType)) {
      rewriteArgTypes.push_back(arg.getType());
      continue;
    }

    arg.setType(*flatType);
    rewriteArgTypes.push_back(*flatType);
    rewriteUsers(arg, memrefType, moduleOp);
  }

  // Flatten local buffers the same way. A statically shaped alloc/alloca has
  // no operands that depend on its shape, so its result can be retyped in
  // place. Their `alignment` attribute is dropped, since downstream consumers
  // (e.g. CIRCT's handshake lowering) do not accept it.
  SmallVector<Value> allocs;
  funcOp.walk([&](Operation *op) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      allocOp.removeAlignmentAttr();
      allocs.push_back(allocOp.getResult());
    } else if (auto allocaOp = dyn_cast<memref::AllocaOp>(op)) {
      allocaOp.removeAlignmentAttr();
      allocs.push_back(allocaOp.getResult());
    }
  });
  for (Value alloc : allocs) {
    auto memrefType = cast<MemRefType>(alloc.getType());
    if (memrefType.getRank() <= 1)
      continue;
    FailureOr<MemRefType> flatType = getFlattenedType(memrefType);
    if (failed(flatType))
      continue;

    alloc.setType(*flatType);
    rewriteUsers(alloc, memrefType, moduleOp);
  }

  // Flatten the remaining multi-dimensional memref arguments passed to calls
  // (direct or indirect) within this function's body, so they match the
  // flattened parameter types this same pass produces for the callee.
  // Functions without a body (e.g. external declarations) are skipped.

  // Collapses `operand` in place to its 1-D form if it's a flattenable
  // multi-dim memref, returning the new type on success.
  auto flattenCallOperand = [&](OpBuilder &callBuilder, Location loc,
                                OpOperand &operand) -> FailureOr<MemRefType> {
    auto memrefType = dyn_cast<MemRefType>(operand.get().getType());
    if (!memrefType || memrefType.getRank() <= 1)
      return failure();

    FailureOr<MemRefType> flatType = getFlattenedType(memrefType);
    if (failed(flatType))
      return failure();

    auto collapseOp = memref::CollapseShapeOp::create(
        callBuilder, loc, *flatType, operand.get(),
        getFullReassociation(memrefType.getRank()));
    operand.set(collapseOp.getResult());
    return flatType;
  };

  // Callees with a body have their results dropped by this same pass (see
  // below), so a call to one must be rebuilt without results. That is only
  // valid if none of the call's results are used.
  auto checkResultsUnused = [](Operation *callOp) -> LogicalResult {
    if (llvm::all_of(callOp->getResults(),
                     [](Value result) { return result.use_empty(); }))
      return success();
    return callOp->emitOpError()
           << "result is used, but the callee's results are dropped by "
              "flatten-contiguous-memref";
  };

  SmallVector<func::CallOp> callOps;
  funcOp.walk([&](func::CallOp callOp) { callOps.push_back(callOp); });
  for (func::CallOp callOp : callOps) {
    if (!callsFlattenedFunc(callOp, moduleOp))
      continue;

    OpBuilder callBuilder(callOp);
    for (OpOperand &operand : callOp.getArgOperandsMutable())
      (void)flattenCallOperand(callBuilder, callOp.getLoc(), operand);

    if (callOp.getNumResults() == 0)
      continue;
    if (failed(checkResultsUnused(callOp)))
      return failure();

    auto newCallOp = func::CallOp::create(callBuilder, callOp.getLoc(),
                                          callOp.getCalleeAttr(), TypeRange{},
                                          callOp.getArgOperands());
    newCallOp->setDiscardableAttrs(callOp->getDiscardableAttrDictionary());
    newCallOp.setArgAttrsAttr(callOp.getArgAttrsAttr());
    callOp.erase();
  }

  // `call_indirect`'s callee is an SSA value of function type, not a
  // symbol, and the op requires (via its `TypesMatchWith` constraints) that
  // argument types exactly match that value's current `FunctionType`. We
  // can only rewrite such a call when the callee value is produced by a
  // `func.constant` referencing a known, non-external function: in that
  // case the constant's result type is flattened in lockstep with the
  // call's operands. Any other indirect callee (a block argument, a value
  // chosen by `arith.select`, loaded from memory, etc.) is left untouched,
  // since we can't tell which concrete function it may refer to.
  SmallVector<func::CallIndirectOp> callIndirectOps;
  funcOp.walk(
      [&](func::CallIndirectOp callOp) { callIndirectOps.push_back(callOp); });
  for (func::CallIndirectOp callOp : callIndirectOps) {
    auto constOp = callOp.getCallee().getDefiningOp<func::ConstantOp>();
    func::FuncOp callee =
        (moduleOp && constOp)
            ? moduleOp.lookupSymbol<func::FuncOp>(constOp.getValueAttr())
            : nullptr;
    if (!callee || callee.isExternal())
      continue;

    auto fnType = cast<FunctionType>(constOp.getResult().getType());
    SmallVector<Type> newInputs(fnType.getInputs());

    OpBuilder callBuilder(callOp);
    for (auto [idx, operand] :
         llvm::enumerate(callOp.getArgOperandsMutable())) {
      FailureOr<MemRefType> flatType =
          flattenCallOperand(callBuilder, callOp.getLoc(), operand);
      if (succeeded(flatType))
        newInputs[idx] = *flatType;
    }

    // The callee's results are dropped as well, so the constant's type
    // loses its results and the call is rebuilt without them.
    if (failed(checkResultsUnused(callOp)))
      return failure();

    if (llvm::ArrayRef(newInputs) != fnType.getInputs() ||
        fnType.getNumResults() != 0) {
      constOp.getResult().setType(
          FunctionType::get(fnType.getContext(), newInputs, TypeRange{}));
    }

    if (callOp.getNumResults() == 0)
      continue;

    auto newCallOp = func::CallIndirectOp::create(
        callBuilder, callOp.getLoc(), callOp.getCallee(),
        callOp.getArgOperands());
    newCallOp->setDiscardableAttrs(callOp->getDiscardableAttrDictionary());
    newCallOp.setArgAttrsAttr(callOp.getArgAttrsAttr());
    callOp.erase();
  }

  // Drop all results: every `func.return` becomes an empty return and the
  // function type loses its results. The returned values are discarded.
  SmallVector<func::ReturnOp> returnOps;
  funcOp.walk([&](func::ReturnOp returnOp) { returnOps.push_back(returnOp); });
  for (func::ReturnOp returnOp : returnOps) {
    if (returnOp.getNumOperands() == 0)
      continue;
    OpBuilder retBuilder(returnOp);
    func::ReturnOp::create(retBuilder, returnOp.getLoc());
    returnOp.erase();
  }

  funcOp.setFunctionType(
      FunctionType::get(funcOp.getContext(), rewriteArgTypes, TypeRange{}));
  funcOp.removeResAttrsAttr();

  return success();
}

struct FlattenContiguousMemref
    : public mlir::memref::impl::FlattenContiguousMemrefPassBase<
          FlattenContiguousMemref> {
  using Base::Base;

  void runOnOperation() override {
    if (failed(flattenContiguousMemref(getOperation())))
      signalPassFailure();
  }
};

} // namespace
