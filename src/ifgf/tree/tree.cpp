#include <ifgf/tree/tree.hpp>

#include <ifgf/tree/morton.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ifgf {

namespace {

// Relative padding of the root cube; keeps the max-face points off the boundary.
constexpr double rootPad = 1e-6;

using Cell = std::array<std::uint32_t, 3>;

struct Builder {
  Tree& t;
  std::vector<Cell> cell;  // integer cell coordinates per box, build only

  std::int32_t addBox(int lvl, std::int32_t parent, std::uint32_t begin, std::uint32_t end,
                      const Cell& c)
  {
    const auto id = t.numBoxes();
    const double h = std::ldexp(t.side, -lvl);
    Vec3d ctr;
    for (int d = 0; d < 3; ++d) ctr[d] = t.rootMin[d] + (static_cast<double>(c[d]) + 0.5) * h;
    t.center.push_back(ctr);
    t.level.push_back(static_cast<std::uint8_t>(lvl));
    t.ptBegin.push_back(begin);
    t.ptEnd.push_back(end);
    t.parent.push_back(parent);
    t.child.push_back({-1, -1, -1, -1, -1, -1, -1, -1});
    t.isLeaf.push_back(1);
    cell.push_back(c);
    return id;
  }
};

}  // namespace

template <class Real>
Tree buildTree(std::span<const Vec3<Real>> points, const TreeParams& params)
{
  if (params.leafSize < 1) throw std::invalid_argument("ifgf::buildTree: leafSize must be >= 1");
  if (params.maxDepth < 0 || params.maxDepth > mortonBits)
    throw std::invalid_argument("ifgf::buildTree: maxDepth must be in [0, 21]");
  if (points.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
    throw std::invalid_argument("ifgf::buildTree: too many points");

  Tree t;
  const auto n = static_cast<std::uint32_t>(points.size());
  if (n == 0) {
    t.levelBegin = {0};
    return t;
  }

  // bounding box in double
  std::vector<Vec3d> pts(n);
  Vec3d lo{+INFINITY, +INFINITY, +INFINITY}, hi{-INFINITY, -INFINITY, -INFINITY};
  for (std::uint32_t i = 0; i < n; ++i) {
    pts[i] = vec3Cast<double>(points[i]);
    for (int d = 0; d < 3; ++d) {
      if (!std::isfinite(pts[i][d])) throw std::invalid_argument("ifgf::buildTree: non-finite point");
      lo[d] = std::min(lo[d], pts[i][d]);
      hi[d] = std::max(hi[d], pts[i][d]);
    }
  }

  // cubic root box centred on the bounding box, padded relative to the extent
  const double extent = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z});
  t.side = extent > 0.0 ? extent * (1.0 + rootPad) : 1.0;
  t.rootMin = 0.5 * (lo + hi) - Vec3d{0.5 * t.side, 0.5 * t.side, 0.5 * t.side};

  // Morton sort (ties broken by index, so the order is deterministic)
  std::vector<std::uint64_t> codes(n);
  for (std::uint32_t i = 0; i < n; ++i) codes[i] = mortonCode(pts[i], t.rootMin, t.side);
  t.perm.resize(n);
  std::iota(t.perm.begin(), t.perm.end(), 0U);
  std::sort(t.perm.begin(), t.perm.end(), [&](std::uint32_t a, std::uint32_t b) {
    return codes[a] != codes[b] ? codes[a] < codes[b] : a < b;
  });
  t.points.resize(n);
  t.codes.resize(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    t.points[i] = pts[t.perm[i]];
    t.codes[i] = codes[t.perm[i]];
  }

  // breadth-first split, level by level; children are appended in parent order and
  // octant order, which keeps every level in Morton order
  Builder b{t, {}};
  t.levelBegin.push_back(0);
  b.addBox(0, -1, 0, n, Cell{0, 0, 0});
  for (int lvl = 0;; ++lvl) {
    t.H.push_back(std::ldexp(t.side, -lvl));
    const std::int32_t b0 = t.levelBegin[lvl], b1 = t.numBoxes();
    for (std::int32_t box = b0; box < b1; ++box) {
      const std::uint32_t begin = t.ptBegin[box], end = t.ptEnd[box];
      if (end - begin <= static_cast<std::uint32_t>(params.leafSize) || lvl >= params.maxDepth)
        continue;
      t.isLeaf[box] = 0;
      // the octant is non-decreasing over the box's sorted codes
      auto pos = t.codes.begin() + begin;
      const auto last = t.codes.begin() + end;
      for (int o = 0; o < 8; ++o) {
        const auto next = std::partition_point(
            pos, last, [&](std::uint64_t c) { return childOctant(c, lvl) <= o; });
        if (next != pos) {
          const Cell pc = b.cell[box];
          const Cell cc{2 * pc[0] + (o & 1U), 2 * pc[1] + ((o >> 1) & 1U), 2 * pc[2] + ((o >> 2) & 1U)};
          const auto id = b.addBox(lvl + 1, box,
                                   static_cast<std::uint32_t>(pos - t.codes.begin()),
                                   static_cast<std::uint32_t>(next - t.codes.begin()), cc);
          t.child[box][o] = id;
        }
        pos = next;
      }
    }
    if (t.numBoxes() == b1) break;
    t.levelBegin.push_back(b1);
  }
  t.levelBegin.push_back(t.numBoxes());
  return t;
}

template Tree buildTree<float>(std::span<const Vec3f>, const TreeParams&);
template Tree buildTree<double>(std::span<const Vec3d>, const TreeParams&);

}  // namespace ifgf
