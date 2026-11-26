#include <ifgf/lists/lists.hpp>

#include <stdexcept>
#include <utility>

namespace ifgf {

namespace {

using Pair = std::pair<std::int32_t, std::int32_t>;  // (target box, source box)

struct Recursion {
  const Tree& S;
  const Tree& T;
  ListParams p;
  std::vector<Pair> near, far;
  std::int64_t cut = 0;

  void visit(std::int32_t s, std::int32_t t)
  {
    const double hs = S.boxSide(s), ht = T.boxSide(t);
    const double dist = exteriorDistance(S.center[s], T.center[t], 0.5 * ht);
    if (dist >= p.alpha * hs) {
      if (dist >= p.cutoffDistance)
        ++cut;
      else
        far.emplace_back(t, s);
      return;
    }
    const bool leafS = S.isLeaf[s] != 0, leafT = T.isLeaf[t] != 0;
    if (leafS && leafT) {
      near.emplace_back(t, s);
      return;
    }
    if (leafS || (!leafT && ht > hs)) {
      for (const auto c : T.child[t])
        if (c >= 0) visit(s, c);
    } else {
      for (const auto c : S.child[s])
        if (c >= 0) visit(c, t);
    }
  }
};

// Stable counting sort by target box; keeps the discovery order within a row.
BoxCsr toCsr(const std::vector<Pair>& pairs, std::int32_t numTargetBoxes)
{
  BoxCsr csr;
  csr.offsets.assign(static_cast<std::size_t>(numTargetBoxes) + 1, 0);
  for (const auto& [t, s] : pairs) ++csr.offsets[t + 1];
  for (std::int32_t i = 0; i < numTargetBoxes; ++i) csr.offsets[i + 1] += csr.offsets[i];
  csr.src.resize(pairs.size());
  std::vector<std::int32_t> fill(csr.offsets.begin(), csr.offsets.end() - 1);
  for (const auto& [t, s] : pairs) csr.src[fill[t]++] = s;
  return csr;
}

}  // namespace

InteractionLists buildLists(const Tree& src, const Tree& tgt, const ListParams& params)
{
  if (!(params.alpha > 0.0)) throw std::invalid_argument("ifgf::buildLists: alpha must be > 0");

  Recursion r{src, tgt, params, {}, {}, 0};
  if (src.numBoxes() > 0 && tgt.numBoxes() > 0) r.visit(0, 0);

  InteractionLists lists;
  lists.near = toCsr(r.near, tgt.numBoxes());
  lists.far = toCsr(r.far, tgt.numBoxes());
  lists.numCutoffPairs = r.cut;
  return lists;
}

}  // namespace ifgf
