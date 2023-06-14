// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <popart/graphcoreoperators.hpp>
#include <popart/names.hpp>
#include <string>
#include <vector>

#include <memory>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/util.hpp>

#include "common.hpp"
#include "group_quantize_decompress.hpp"
#include "group_quantize_decompressx.hpp"

#include <iostream>
namespace popart {

/////////////////////////////////////////////////////////////
////// Fwd op

GroupQuantizeDecompressOp::GroupQuantizeDecompressOp(
    const OperatorIdentifier &_opid, const Op::Settings &settings_, const std::string& rootPath_)
    : Op(_opid, settings_), rootPath(rootPath_){}

std::unique_ptr<Op> GroupQuantizeDecompressOp::clone() const {
  return std::make_unique<GroupQuantizeDecompressOp>(*this);
}

void GroupQuantizeDecompressOp::setup() {
  auto xInfo = inInfo(0);
  auto groupScaleInfo = inInfo(1);
  auto groupBiasInfo = inInfo(2);

  // check expected shapes
  if (xInfo.rank() != 3) {
    throw error("GroupQuantizeDecompressOp::setup x should have rank 3");
  }

  if (groupScaleInfo.rank() != xInfo.rank()) {
    throw error(
        "GroupQuantizeDecompressOp::setup groupScale should same rank as x");
  }

  if (groupBiasInfo.rank() != xInfo.rank()) {
    throw error(
        "GroupQuantizeDecompressOp::setup groupBias should same rank as x");
  }

  if (groupScaleInfo.shape()[2] != 1) {
    throw error("GroupQuantizeDecompressOp::setup groupScale shape at last "
                "dimension should be 1");
  }

  if (groupBiasInfo.shape()[2] != 1) {
    throw error("GroupQuantizeDecompressOp::setup groupBias shape at last "
                "dimension should be 1");
  }

  if (groupScaleInfo.shape() != groupScaleInfo.shape()) {
    throw error("GroupQuantizeDecompressOp::setup groupScale and groupBias "
                "should have same shape");
  }

  auto nRows = xInfo.shape()[0];
  auto nGroups = xInfo.shape()[1];
  auto nIds = xInfo.shape()[2];

  // x decompressed
  outInfo(0) = TensorInfo(groupScaleInfo.data_type(),
                          Shape{nRows, nGroups * nIds * 4});
}

void GroupQuantizeDecompressOp::appendOutlineAttributes(
    OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
}


using popart::DataType;
using popart::OpDefinition;


// const popart::OperatorIdentifier GroupQuantizeDecompressOpId = {"optimum.custom_ops", "GroupQuantizeDecompressOp", 1};
static OpDefinition::DataTypes INT16_T = {DataType::INT16};
static OpDefinition::DataTypes FLOAT_T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition GroupQuantizeDecompressOpDef({OpDefinition::Inputs({{"xInfo", INT16_T}, {"groupScale", FLOAT_T}, {"groupBias", FLOAT_T}}),
					  OpDefinition::Outputs({{"output", FLOAT_T}}),
					  OpDefinition::Attributes()});

static popart::OpCreator<GroupQuantizeDecompressOp> GroupQuantizeDecompressOpCreator(
popart::OpDefinitions({{GroupQuantizeDecompressId, GroupQuantizeDecompressOpDef}}),
[](const popart::OpCreatorInfo& info) {
 auto rootPath = info.attributes.getAttribute<popart::Attributes::String>("root_path");
 return std::make_unique<GroupQuantizeDecompressOp>(info.opid, info.settings, rootPath);
},
true);

// static popart::popx::OpxCreator<GroupQuantizeDecompressOpx> GroupQuantizeDecompressOpxCreator({GroupQuantizeDecompressOpId});

} // namespace popart
