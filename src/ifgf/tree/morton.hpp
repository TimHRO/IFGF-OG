#pragma once

// 63-bit Morton codes (21 bits per axis). Within every 3-bit group, bit d belongs
// to axis d, so the group read at a level is the child octant: bit d set = upper
// half in axis d.

#include <ifgf/core/vec3.hpp>

#include <algorithm>
#include <cstdint>

namespace ifgf {

inline constexpr int mortonBits = 21;  // per axis; also the maximal tree depth

// Spread the lower 21 bits of v so that they occupy every third bit.
constexpr std::uint64_t spreadBits(std::uint64_t v)
{
  v &= 0x1fffffULL;
  v = (v | v << 32) & 0x1f00000000ffffULL;
  v = (v | v << 16) & 0x1f0000ff0000ffULL;
  v = (v | v << 8) & 0x100f00f00f00f00fULL;
  v = (v | v << 4) & 0x10c30c30c30c30c3ULL;
  v = (v | v << 2) & 0x1249249249249249ULL;
  return v;
}

// Morton code of p in the cube [rootMin, rootMin + side]^3; points outside are clamped.
inline std::uint64_t mortonCode(const Vec3d& p, const Vec3d& rootMin, double side)
{
  constexpr double scale = static_cast<double>(std::uint64_t{1} << mortonBits);
  constexpr std::uint64_t qmax = (std::uint64_t{1} << mortonBits) - 1;
  std::uint64_t code = 0;
  for (int d = 0; d < 3; ++d) {
    const double t = std::clamp((p[d] - rootMin[d]) / side, 0.0, 1.0);
    const std::uint64_t q = std::min(static_cast<std::uint64_t>(t * scale), qmax);
    code |= spreadBits(q) << d;
  }
  return code;
}

// Octant (0..7) of the child of a level-`level` box that contains `code`.
// Valid for 0 <= level < mortonBits.
constexpr int childOctant(std::uint64_t code, int level)
{
  return static_cast<int>((code >> (3 * (mortonBits - 1 - level))) & 7U);
}

}  // namespace ifgf
