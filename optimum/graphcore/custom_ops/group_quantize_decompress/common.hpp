// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#ifndef GUARD_GROUPQUANTIZEDECOMPRESS_OPIDS
#define GUARD_GROUPQUANTIZEDECOMPRESS_OPIDS

#include <popart/attributes.hpp>
#include <popart/error.hpp>
#include <popart/names.hpp>
#include <popart/operatoridentifier.hpp>

using InMapType = std::map<popart::InIndex, popart::TensorId>;
using OutMapType = std::map<popart::OutIndex, popart::TensorId>;
using OutIndex = int;

namespace popart {

#define CUSTOM_OP_DOMAIN "optimum.custom_ops"

const popart::OperatorIdentifier GroupQuantizeDecompressId = OperatorIdentifier{
    CUSTOM_OP_DOMAIN,
    "GroupQuantizeDecompress",
    1,      // Op version
    {3, 3}, // number of inputs
    1       // number of outputs
};

} // namespace popart

#endif
