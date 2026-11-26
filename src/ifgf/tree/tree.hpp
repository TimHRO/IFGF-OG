#pragma once

// Host octree over cubic boxes.
//
// All boxes live in flat arrays, level-major (breadth-first): the boxes of level l
// are [levelBegin[l], levelBegin[l+1]). Within a level, boxes are in Morton order,
// so their point ranges are increasing and disjoint. Geometry is double.

#include <ifgf/core/vec3.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace ifgf {

struct TreeParams {
  int leafSize = 64;  // a box with more points than this is split ...
  int maxDepth = 21;  // ... unless it is on this level (at most mortonBits)
};

struct Tree {
  // root cube [rootMin, rootMin + side]^3
  Vec3d rootMin{};
  double side = 0.0;

  // levels
  std::vector<std::int32_t> levelBegin;  // numLevels() + 1 entries
  std::vector<double> H;                 // side length per level, side / 2^l

  // boxes
  std::vector<Vec3d> center;
  std::vector<std::uint8_t> level;
  std::vector<std::uint32_t> ptBegin, ptEnd;  // range in the sorted points
  std::vector<std::int32_t> parent;           // -1 for the root
  std::vector<std::array<std::int32_t, 8>> child;  // octant slots, -1 = no child
  std::vector<std::uint8_t> isLeaf;

  // points, Morton-sorted
  std::vector<Vec3d> points;
  std::vector<std::uint64_t> codes;
  std::vector<std::uint32_t> perm;  // sorted index -> original index

  int numLevels() const { return static_cast<int>(levelBegin.size()) - 1; }
  std::int32_t numBoxes() const { return static_cast<std::int32_t>(center.size()); }
  std::uint32_t numPoints() const { return static_cast<std::uint32_t>(points.size()); }
  std::uint32_t numPoints(std::int32_t box) const { return ptEnd[box] - ptBegin[box]; }
  double boxSide(std::int32_t box) const { return H[level[box]]; }
};

// Builds the tree. Throws std::invalid_argument for bad params or non-finite points.
template <class Real>
Tree buildTree(std::span<const Vec3<Real>> points, const TreeParams& params);

extern template Tree buildTree<float>(std::span<const Vec3f>, const TreeParams&);
extern template Tree buildTree<double>(std::span<const Vec3d>, const TreeParams&);

}  // namespace ifgf
