// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <ipu_intrinsics>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

template <class R, class T> R as(T x) { return *reinterpret_cast<R *>(&x); }

namespace {

// Converts x containing 4 packed 4-bit ints to half4
//    x = [aaaa|bbbb|cccc|dddd]
//    y = half4{a, b, c, d} // in range [0, 16]
//
half4 convert4(short x) {
  // copy of x shifted by 8 bits
  auto x_shr = x >> 8;
  // reinterpret uint16 values as float16 (will be in subnormal range)
  auto z = half4{as<half>(x_shr), as<half>(x_shr), as<half>(x), as<half>(x)};
  // create mask for andc (and-complement) - better for vectorisation
  auto mask = ushort4{0xff0fu, 0xfff0u, 0xff0fu, 0xfff0u};
  // apply mask with andc
  auto y = as<half4>(ipu::andc(as<float2>(z), as<float2>(mask)));
  // scale from subnormal to normal range
  // split into two multiplications to not exceed max fp16 value (~2^16)
  const auto scale4 = half4{2048, 32768, 2048, 32768};
  const auto scale1 = half{512};
  return (y * scale4) * scale1;
}

} // namespace

// Convert a vector of packed 4-bit ints to half4
//
void convert4_vec(const short *x, half4 *y, unsigned n) {
  for (auto i = 0u; i < n; ++i) {
    y[i] = convert4(x[i]);
  }
}

struct DecompressPacked4BitTensor : poplar::Vertex {
  poplar::Input<poplar::Vector<int16_t, poplar::VectorLayout::SPAN>> input;
  poplar::Output<poplar::Vector<half, poplar::VectorLayout::ONE_PTR, 8>> output;

  bool compute() {
    convert4_vec(&input[0], reinterpret_cast<half4 *>(&output[0]),
                 input.size());
    return true;
  }
};
