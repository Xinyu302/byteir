#include "byteir/Dialect/GPU/Passes.h"
#include "byteir/Dialect/GPU/Transforms/LegalizeGPULaunch.h"
#include "byteir/Dialect/GPU/Transforms/Transforms.h"
#include "byteir/Dialect/GPU/Transforms/Utils.h"
#include "byteir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include <string>

#include "PassDetail.h"

using namespace llvm;

namespace mlir {
#define GEN_PASS_DEF_GPULOOPINVARIANTCODEMOTION
#include "byteir/Dialect/GPU/Passes.h.inc"
namespace {
static void moveLoopInvariantCodeFromGuaranteedLoops(Operation *target) {
  // Walk through all loops in a function in innermost-loop-first order. This
  // way, we first LICM from the inner loop, and place the ops in
  // the outer loop, which in turn can be further LICM'ed.
  //
  // Hoisting is only performed on loops with guaranteed non-zero trip counts.
  // `scf.forall` ops with mapping attributes can never be proven to have a
  // non-zero trip count until the loop is resolved and is blanket included
  // here.
  target->walk([&](LoopLikeOpInterface loopLike) {
    if (auto forallOp = dyn_cast<scf::ForallOp>(*loopLike)) {
      if (forallOp.getMapping()) {
        return;
      }
    }

    // Skip loops without lower/upper bounds. There is no generic way to verify
    // whether a loop has at least one trip so new loop types of interest can be
    // added as needed. For example, `scf.while` needs non-trivial analysis of
    // its condition region to know that it has at least one trip.
    /*
    std::optional<SmallVector<OpFoldResult>> maybeLowerBounds =
        loopLike.getLoopLowerBounds();
    std::optional<SmallVector<OpFoldResult>> maybeUpperBounds =
        loopLike.getLoopUpperBounds();
    std::optional<SmallVector<Value>> maybeIvs =
        loopLike.getLoopInductionVars();
    */
    std::optional<OpFoldResult> maybeLowerBound =
        loopLike.getSingleLowerBound();
    std::optional<OpFoldResult> maybeUpperBound =
        loopLike.getSingleUpperBound();
    std::optional<Value> maybeIv = loopLike.getSingleInductionVar();
    if (!maybeLowerBound || !maybeUpperBound || !maybeIv) {
      return;
    }

    // If any lower + upper bound pair cannot be definitely verified as lb < ub
    // then the loop may have a zero trip count.
    auto lb = *maybeLowerBound;
    auto ub = *maybeUpperBound;
    auto iv = *maybeIv;
    if (iv.getType().isIndex()) {
      if (!ValueBoundsConstraintSet::compare(lb, ValueBoundsConstraintSet::LT,
                                             ub)) {
        return;
      }
    } else {
      // Weaker test for non-`index` operands to some loops
      // like scf.for, since the value bounds interface requires index types.
      auto maybeLb = getConstantIntValue(lb);
      auto maybeUb = getConstantIntValue(ub);
      if (!maybeLb || !maybeUb)
        return;
      if (*maybeLb >= *maybeUb)
        return;
    }

    moveLoopInvariantCode(loopLike);
  });
}

/// IREE loop invariant code motion (LICM) pass.
struct GPULoopInvariantCodeMotionPass
    : public impl::GPULoopInvariantCodeMotionBase<
          GPULoopInvariantCodeMotionPass> {
  void runOnOperation() override;
};
} // namespace

void GPULoopInvariantCodeMotionPass::runOnOperation() {
  moveLoopInvariantCodeFromGuaranteedLoops(getOperation());
}
} // namespace mlir
