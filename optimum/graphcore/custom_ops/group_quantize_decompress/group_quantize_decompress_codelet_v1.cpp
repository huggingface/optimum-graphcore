// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include <ipu_intrinsics>
#include <ipu_memory_intrinsics>
#include <ipu_vector_math>

#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

template <class R, class T> R as(T x) { return *reinterpret_cast<R *>(&x); }

namespace {

/**
 * @brief Converts x containing 4 packed 4-bit ints to half4.
 *
 *    x = [aaaa|bbbb|cccc|dddd]
 *    y = half4{a, b, c, d} // in range [0, 16]
 * Original implementation.
 */
inline half4 from_int4v4_to_halfv4_convert_v0(unsigned short x) {
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

// C
//
/**
 * @brief Convert vectors of int4 to half vectors.
 * Semi-optimized version! Re-organizing the packing compared to v0 in order to
 * do more efficient vectorized decoding.
 */
inline void from_int4v4_to_halfv4_convert_v1(const unsigned short *x, half4 *y,
                                             unsigned n) {
  const uint16_t size = uint16_t(n) / 4;
  auto input_ptr = reinterpret_cast<const half4 *>(x);

  half4 out_fp16v4, inmaskv4, in_int4v160, in_int4v161, in_int4v16;
  half4 out0_fp16v4, out1_fp16v4;
  const auto mask0 = ushort4{0xfff0u, 0xfff0u, 0xfff0u, 0xfff0u};
  const auto mask1 = ushort4{0xff0fu, 0xff0fu, 0xff0fu, 0xff0fu};
  const auto scale1 = half{512};

  for (uint16_t i = 0; i < size; ++i) {
    // int4v16 single load.
    in_int4v16 = ipu::load_postinc(&input_ptr, 1);

    // Unroll decoding of 4xv4 half vectors.
    inmaskv4 = as<half4>(ipu::andc(as<float2>(in_int4v16), as<float2>(mask0)));
    out0_fp16v4 = (inmaskv4 * half{32768}) * scale1;
    ipu::store_postinc(&y, out0_fp16v4, 1);

    inmaskv4 = as<half4>(ipu::andc(as<float2>(in_int4v16), as<float2>(mask1)));
    out1_fp16v4 = (inmaskv4 * half{2048}) * scale1;
    ipu::store_postinc(&y, out1_fp16v4, 1);

    auto x_shr = as<uint64_t>(in_int4v16) >> 8;

    inmaskv4 = as<half4>(ipu::andc(as<float2>(x_shr), as<float2>(mask0)));
    out_fp16v4 = (inmaskv4 * half{32768}) * scale1;
    ipu::store_postinc(&y, out_fp16v4, 1);

    inmaskv4 = as<half4>(ipu::andc(as<float2>(x_shr), as<float2>(mask1)));
    out_fp16v4 = (inmaskv4 * half{2048}) * scale1;
    ipu::store_postinc(&y, out_fp16v4, 1);
  }
}

/**
 * IPU unpack vertex, from int4 to float16 (version 0).
 */
struct DecompressPacked4BitTensorV0 : poplar::Vertex {
  poplar::Input<poplar::Vector<int16_t, poplar::VectorLayout::SPAN, 8>> input;
  poplar::Output<poplar::Vector<half, poplar::VectorLayout::SPAN, 8>> output;

  bool compute() {
    const uint16_t insize = input.size();
    auto in_ptr = reinterpret_cast<const unsigned short *>(&input[0]);
    auto out_ptr = reinterpret_cast<half4 *>(&output[0]);

    for (uint16_t i = 0; i < insize; ++i) {
      out_ptr[i] = from_int4v4_to_halfv4_convert_v0(in_ptr[i]);
    }
    return true;
  }
};

/**
 * IPU unpack vertex, from int4 to float16 (version 1).
 */
struct DecompressPacked4BitTensorV1 : poplar::Vertex {
  poplar::Input<poplar::Vector<int16_t, poplar::VectorLayout::SPAN, 8>> input;
  poplar::Output<poplar::Vector<half, poplar::VectorLayout::SPAN, 8>> output;

  bool compute() {
    from_int4v4_to_halfv4_convert_v1(
        reinterpret_cast<const unsigned short *>(&input[0]),
        reinterpret_cast<half4 *>(&output[0]), input.size());
    return true;
  }
};
