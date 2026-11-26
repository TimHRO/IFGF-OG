#pragma once

// Near/far interaction lists between a source and a target tree.

#include <ifgf/core/vec3.hpp>
#include <ifgf/tree/tree.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace ifgf {

struct ListParams {
  double alpha = std::sqrt(3.0);  // admissible iff dist(T, c_S) >= alpha * H_S
  double cutoffDistance = std::numeric_limits<double>::infinity();  // drop admissible pairs with dist >= this
};

// Source boxes per target box, compressed rows. Target boxes are tree-global ids,
// so the rows of target level l are [levelBegin[l], levelBegin[l+1]).
struct BoxCsr {
  std::vector<std::int32_t> offsets;  // numTargetBoxes + 1
  std::vector<std::int32_t> src;      // source box ids

  std::int64_t numPairs() const { return static_cast<std::int64_t>(src.size()); }
};

struct InteractionLists {
  BoxCsr near;  // leaf x leaf, direct evaluation
  BoxCsr far;   // admissible, evaluated through the source box's cones
  std::int64_t numCutoffPairs = 0;  // admissible pairs dropped by cutoffDistance
};

// Distance from p to the cube with the given center and half side (0 inside).
inline double exteriorDistance(const Vec3d& p, const Vec3d& cubeCenter, double halfSide)
{
  double s = 0.0;
  for (int d = 0; d < 3; ++d) {
    const double e = std::abs(p[d] - cubeCenter[d]) - halfSide;
    if (e > 0.0) s += e * e;
  }
  return std::sqrt(s);
}

// Dual-tree recursion from (root_S, root_T):
//   1. admissible -> far (or dropped by the cutoff)
//   2. both leaves -> near
//   3. exactly one leaf -> recurse into the other
//   4. otherwise recurse into the larger box; on a tie into the source
InteractionLists buildLists(const Tree& src, const Tree& tgt, const ListParams& params);

}  // namespace ifgf
