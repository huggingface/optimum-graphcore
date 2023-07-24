// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
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
