//===- GemmCodegen.cpp ---------------------------------------*--- C++ -*-===//
//
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/GPU/GemmCodegen.h"
#include "byteir/Conversion/GemmCodeGen/Utils/GPUCodeGenUtils.h"
#include "byteir/Conversion/ToGPU/ToGPU.h"
#include "byteir/Conversion/ToLLVM/ToLLVM.h"
#include "byteir/Dialect/Linalg/TransformOps/LinalgExtTransformOps.h"
#include "byteir/Dialect/Linalg/Transforms/LinalgGPUPassCommon.h"
#include "byteir/Dialect/Linalg/Transforms/LinalgPrefetch.h"
#include "byteir/Dialect/Transform/IR/TransformExtOps.h"
#include "byteir/Dialect/Transform/Transforms/TransformInsertion.h"
#include "byteir/Pipelines/Common/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallSet.h"

#include <optional>

using namespace mlir;

namespace {

/// copy from ReductionCodegen.cpp. Should make it to a util.

struct ProducerSelector {
  uint64_t operandNumber;
  llvm::StringRef opName;
  std::vector<ProducerSelector> producerSelectors;

  ProducerSelector(uint64_t operandNumber, llvm::StringRef opName)
      : operandNumber(operandNumber), opName(opName) {}

  static bool detectFillOperand(OpOperand *opOperand,
                                std::vector<ProducerSelector> &selectors) {
    if (opOperand->get().getDefiningOp<linalg::FillOp>()) {
      selectors.emplace_back(opOperand->getOperandNumber(),
                             linalg::FillOp::getOperationName());
      return true;
    }
    return false;
  }

  static bool detectPadOperand(OpOperand *opOperand,
                               std::vector<ProducerSelector> &selectors) {
    Operation *definingOp = opOperand->get().getDefiningOp();
    if (!definingOp)
      return false;

    if (llvm::isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(definingOp)) {
      ProducerSelector selector(opOperand->getOperandNumber(),
                                definingOp->getName().getStringRef());
      if (detectPadOperand(&definingOp->getOpOperand(0),
                           selector.producerSelectors)) {
        selectors.emplace_back(std::move(selector));
        return true;
      }
    } else if (llvm::isa<tensor::PadOp>(definingOp)) {
      selectors.emplace_back(opOperand->getOperandNumber(),
                             tensor::PadOp::getOperationName());
      return true;
    }
    return false;
  }
};

struct GridTileConfig {
  SmallVector<int64_t, 3> tileSizes;
  std::vector<ProducerSelector> fuseCandidates;
};

std::optional<GridTileConfig>
getGridTileConfig(linalg::LinalgOp linalgOp,
                  SmallVector<int64_t, 3> tileSizes) {
  if (!llvm::isa<linalg::MatmulOp>(linalgOp))
    return std::nullopt;

  std::vector<ProducerSelector> fuseCandidates;
  for (OpOperand &opOperand : linalgOp.getDpsInitsMutable()) {
    ProducerSelector::detectFillOperand(&opOperand, fuseCandidates);
  }

  return GridTileConfig{tileSizes, fuseCandidates};
}

void processProducerSelectors(
    ImplicitLocOpBuilder &b,
    const std::vector<ProducerSelector> &producerSelectors, Value fuseInto,
    SmallVector<Value> &selected, Type producerType = nullptr) {
  for (auto selector : producerSelectors) {
    auto producer = b.create<transform::GetProducerOfOperand>(
        /* producer type */ producerType
            ? producerType
            : transform::OperationType::get(b.getContext(), selector.opName),
        /* target */ fuseInto,
        /* operand number */ selector.operandNumber);
    selected.push_back(producer.getProducer());
    processProducerSelectors(b, selector.producerSelectors, selected.back(),
                             selected);
  }
}

transform::TileUsingForallOp
tileToForallAndFuseImpl(ImplicitLocOpBuilder &b, Value toTile,
                        const SmallVector<int64_t> &tileSizes,
                        const SmallVector<Attribute> &mapping,
                        const std::vector<ProducerSelector> &fuseCandidates) {
  SmallVector<Value> toBeFused;
  processProducerSelectors(b, fuseCandidates, toTile, toBeFused);

  auto tileOp = b.create<transform::TileUsingForallOp>(
      /* target */ toTile,
      /* staticTileSizes */ tileSizes,
      /* ctor tag */ transform::TileSizesSpec(),
      /* mapping */ b.getArrayAttr(mapping));
  for (auto &&producerOp : toBeFused) {
    b.create<transform::FuseIntoContainingOp>(
        /* producerOp */ producerOp,
        /* containingOp */ tileOp.getForallOp());
  }
  return tileOp;
}

void createGPUTileGemmTransformImpl(OpPassManager &pm,
                                    const std::string &anchor,
                                    const std::string &prefix) {
  TransformInsertionConfig config;
  config.funcAnchor = anchor;
  config.matchPrefix = prefix;
  config.opFilter = [=](Operation *op) {
    if (auto linalgOp = llvm::dyn_cast_or_null<linalg::LinalgOp>(op)) {
      func::FuncOp funcOp = op->getParentOfType<func::FuncOp>();
      SmallVector<int64_t, 3> tileSizeConfig = getGemmTileSize(funcOp).value();

      return getGridTileConfig(linalgOp, tileSizeConfig).has_value();
    }
    return false;
  };

  config.transformBuilder = [=](ImplicitLocOpBuilder &b, Operation *op,
                                Value pdlV) {
    func::FuncOp funcOp = op->getParentOfType<func::FuncOp>();
    SmallVector<int64_t, 3> tileSizeConfig = getGemmTileSize(funcOp).value();
    SmallVector<int64_t, 3> workgroupSize = getGemmBlockSize(funcOp).value();
    int64_t stages = getGemmPipelineDepth(funcOp).value();

    auto gridTileConfig =
        getGridTileConfig(llvm::cast<linalg::LinalgOp>(op), tileSizeConfig)
            .value();

    Value block_idx_y = b.create<transform::ParamConstantOp>(
        /* type */ pdl::AttributeType::get(b.getContext()),
        /* value */ b.getStringAttr("block_id.y"));

    Value block_idx_x = b.create<transform::ParamConstantOp>(
        /* type */ pdl::AttributeType::get(b.getContext()),
        /* value */ b.getStringAttr("block_id.x"));

    Value mmaLevel = b.create<transform::ParamConstantOp>(
        /* type */ pdl::AttributeType::get(b.getContext()),
        /* value */ b.getStringAttr("Threadblock"));
    Value target = b.create<transform::ParamConstantOp>(
        /* type */ pdl::AttributeType::get(b.getContext()),
        /* value */ b.getStringAttr("nv_sm_80"));

    Value stagesParam = b.create<transform::ParamConstantOp>(
        /* type */ pdl::AttributeType::get(b.getContext()),
        /* value */ b.getI64IntegerAttr(stages));

    auto mapping =
        llvm::to_vector(llvm::map_range(SmallVector{1, 0}, [](int64_t i) {
          return static_cast<gpu::MappingId>(i);
        }));
    auto mappingAttrs = llvm::to_vector(
        llvm::map_range(mapping, [&](gpu::MappingId dim) -> Attribute {
          return gpu::GPUBlockMappingAttr::get(b.getContext(), dim);
        }));

    auto tileMatmulOp = tileToForallAndFuseImpl(
        b, pdlV, SmallVector{tileSizeConfig[0], tileSizeConfig[1]},
        mappingAttrs, gridTileConfig.fuseCandidates);

    pdlV = tileMatmulOp.getTiledOp();
    auto tileKMatmulOp = b.create<transform::TileUsingForOp>(
        pdlV, SmallVector<int64_t>{0, 0, tileSizeConfig[2]});
    pdlV = tileKMatmulOp.getTiledLinalgOp();

    b.create<transform::AnnotateOp>(pdlV, getLinalgMMALevelAttrName(),
                                    mmaLevel);
    b.create<transform::AnnotateOp>(pdlV, getLinalgTargetAttrName(), target);
    b.create<transform::AnnotateOp>(pdlV, getMMAPatternAttrName(), Value());
  };

  pm.addPass(createGenericTransformInsertionPass(config));
}

void createPrepareLinalgPrefetchTransformImpl(OpPassManager &pm,
                                              const std::string &anchor,
                                              const std::string &prefix) {
  TransformInsertionConfig config;
  config.funcAnchor = anchor;
  config.matchPrefix = prefix;

  config.opFilter = [=](Operation *op) {
    if (llvm::dyn_cast<linalg::CopyOp>(op)) {
      if (op->hasAttr(getLinalgLoadMatrixBAttrName()) ||
          op->hasAttr(getLinalgLoadMatrixAAttrName())) {
        return true;
      }
      return false;
    }
    return false;
  };

  config.transformBuilder = [=](ImplicitLocOpBuilder &b, Operation *op,
                                Value pdlV) {
    func::FuncOp funcOp = op->getParentOfType<func::FuncOp>();
    int64_t stages = getGemmPipelineDepth(funcOp).value();

    Value stageValue = b.create<transform::ParamConstantOp>(
        /* type */ pdl::AttributeType::get(b.getContext()),
        /* value */ b.getI64IntegerAttr(stages));
    b.create<transform::AnnotateOp>(pdlV, getLinalgCopyAsyncAttrName(),
                                    Value());
    b.create<transform::AnnotateOp>(pdlV, getPrefetchAttrName(), Value());
    b.create<transform::AnnotateOp>(pdlV, getPrefetchStagesAttrName(),
                                    stageValue);
  };

  pm.addPass(createGenericTransformInsertionPass(config));
}

} // namespace

void mlir::createGPUTileGemmTransform(OpPassManager &pm,
                                      const GPUTileGemmOptions &options) {
  invokeOpPassPipelineBuilder(createGPUTileGemmTransformImpl, pm,
                              options.funcAnchor, options.annotatePrefix);
}

void mlir::createPrepareLinalgPrefetchTransform(
    OpPassManager &pm, const GPUTileGemmOptions &options) {
  invokeOpPassPipelineBuilder(createPrepareLinalgPrefetchTransformImpl, pm,
                              options.funcAnchor, options.annotatePrefix);
}
