// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
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