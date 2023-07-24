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
#ifndef GUARD_GROUPQUANTIZEDECOMPRESS_HPP
#define GUARD_GROUPQUANTIZEDECOMPRESS_HPP

#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/vendored/optional.hpp>

#include "common.hpp"

namespace popart {

class GroupQuantizeDecompressOp : public Op {
public:
  std::string rootPath;
  GroupQuantizeDecompressOp(const OperatorIdentifier &_opid,
                            const Op::Settings &settings_,
                            const std::string &rootPath_);

  std::unique_ptr<Op> clone() const override;
  void setup() final;

  float getSubgraphValue() const override { return getHighSubgraphValue(); }

  // static GroupQuantizeDecompressOp *
  // createOpInGraph(popart::Graph &graph, const InMapType &in,
  //                 const OutMapType &out, const popart::Op::Settings &settings) {
  //   return graph.createConnectedOp<GroupQuantizeDecompressOp>(
  //       in, out, GroupQuantizeDecompressId, settings);
  // }

  void appendOutlineAttributes(OpSerialiserBase &) const override;
};

} // namespace popart

#endif