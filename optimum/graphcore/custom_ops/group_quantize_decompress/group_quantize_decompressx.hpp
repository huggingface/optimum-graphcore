// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GROUPQUANTIZEDECOMPRESSX_HPP
#define GUARD_NEURALNET_GROUPQUANTIZEDECOMPRESSX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>
#include <vector>

namespace popart {
namespace popx {

class GroupQuantizeDecompressOpx : public Opx {
public:
  GroupQuantizeDecompressOpx(Op *, Devicex *);

  void grow(poplar::program::Sequence &) const;
};

} // namespace popx
} // namespace popart

#endif