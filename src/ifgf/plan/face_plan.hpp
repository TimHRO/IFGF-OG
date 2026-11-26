#pragma once

// FacePlan<Kernel, P, N0>: the IFGF operator with face-chart cones (cones/face_cones.hpp), FP32 on one device.
//
// init (once): octrees, interaction lists, active cones and level maps (on the device), upward plan, upload.
// apply(weights), in one in-order queue:
//   1. near field  group per target box with near pairs; direct sum with compensated FP32 geometry
//   2. leaf cones  group per (leaf box, active cone): node values = sum of w * factored kernel at the face-chart
//                  nodes (one thread per ray of radial nodes), then the transform -> Chebyshev coefficients
//   3. upward      per parent level, fine to coarse; work-group = a pack of up to UPCAP active elements of one
//                  (parent box, face) with their node values in local memory. Per staged child cone two kinds of
//                  tasks run on 16 half-warps, both starting with a radial contraction:
//                  - structured (element, layer, u-node range, v-node range): u and v contractions
//                  - fallback lines: parent nodes that leave the child's chart, evaluated in the neighbouring face
//                  then the transfer factor. Each node belongs to exactly one task (no atomics). At the end the
//                  group transforms its elements' values to coefficients in local memory and writes them once.
//   4. far field   per chunk of targets: lookup of the target's cone in each far source box, interpolation, times
//                  the centred factor (compensated FP32 phase)

#include <ifgf/cones/face_activation_device.hpp>
#include <ifgf/cones/face_cones.hpp>
#include <ifgf/core/compensated.hpp>
#include <ifgf/exec/device_buffer.hpp>
#include <ifgf/interp/eval_kernel.hpp>
#include <ifgf/interp/tables.hpp>
#include <ifgf/interp/transform_kernel.hpp>
#include <ifgf/kernels/helmholtz.hpp>
#include <ifgf/lists/lists.hpp>
#include <ifgf/tree/tree.hpp>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <complex>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>


namespace ifgf {

struct FacePlanOptions {
  int leafSize = 64;
  double alpha = std::sqrt(3.0);
  double selfTolRel = 1e-6;
  double coneC = 0.15;            // density = coneC |kappa|; today's SL setting
  cones::FaceParams face{};       // grid factors, minimum grid, chart extension delta, radial range
};

struct FacePlanStats {
  double buildMs = 0, uploadMs = 0;                       // init: host build (trees, lists, cones), upload
  double treeMs = 0, listsMs = 0, conesMs = 0;            // parts of buildMs
  double farActivationMs = 0, parentActivationMs = 0, levelMapMs = 0, upPlanMs = 0;
  double nearMs = 0, leafMs = 0, upwardMs = 0, farMs = 0, applyMs = 0;  // last apply
  std::vector<double> upwardLevelMs;                      // per parent level
  std::int64_t nearPairs = 0, farPairs = 0, cones = 0;
  int lookupMisses = 0;                                   // far-field lookups of an inactive cone (must stay 0)
  std::size_t deviceBytes = 0;                            // device memory held by the plan after init
  std::size_t initPeakDeviceBytes = 0;                    // peak device memory during init
  std::vector<std::int64_t> conesPerLevel;                // active cones per source level
};

namespace face_detail {

// index of el in the sorted range [lo, hi) of a, or -1
IFGF_INLINE std::int32_t findSlot(const std::int32_t* a, std::int32_t lo, std::int32_t hi, std::int32_t el)
{
  const std::int32_t end = hi;
  while (lo < hi) {
    const std::int32_t mid = (lo + hi) / 2;
    if (a[mid] < el)
      lo = mid + 1;
    else
      hi = mid;
  }
  return (lo < end && a[lo] == el) ? lo : -1;
}

constexpr int lineShift = 28;  // line task: element << 28 | line index within the level

struct alignas(16) UpGroup {
  std::int32_t face, n, stBegin, stEnd;
};
struct alignas(16) UpStaged {
  std::int32_t oct, slot, iuc, ivc;
  std::int32_t tBegin, tEnd;        // structured tasks
  float wC, wHinv;                  // child element's w center and 1 / half width
  std::int32_t lBegin, lEnd, pad0, pad1;  // fallback line tasks (in the same task array)
};
// near pair with the box offset as a hi/lo pair (compensated near field)
struct alignas(16) NearPairC {
  Vec3f d, dl;               // c_S - c_T = d + dl
  std::uint32_t s0, n, cum;  // sources [s0, s0 + n); sum of n over the earlier pairs of t
};

template <class Real>
struct FarPairF {
  cones::FaceGrid<Real> grid;
  Vec3<Real> center;
  Real H;
  std::int32_t coneBegin, coneEnd;
};

IFGF_INLINE std::uint64_t spread3(std::uint32_t v)
{
  std::uint64_t x = v & 0x1fffff;
  x = (x | x << 32) & 0x1f00000000ffffULL;
  x = (x | x << 16) & 0x1f0000ff0000ffULL;
  x = (x | x << 8) & 0x100f00f00f00f00fULL;
  x = (x | x << 4) & 0x10c30c30c30c30c3ULL;
  x = (x | x << 2) & 0x1249249249249249ULL;
  return x;
}

// staged child cone: w centre and 1 / half width of child element (iwc, iuc, ivc); hw and 1 / (hw / 2) are computed
// on the host, the rest is exact IEEE add/mul without contraction
template <class Real>
IFGF_INLINE Real stagedWCenter(Real wlo, int iwc, Real hw)
{
  IFGF_NO_FP_CONTRACT
  return wlo + (Real(iwc) + Real(0.5)) * hw;
}
template <class Real>
inline void stagedW(const cones::FaceGrid<Real>& gc, int iwc, int iuc, int ivc, float& wC, float& wHinv)
{
  IFGF_NO_FP_CONTRACT
  const Real hw = gc.hw(iuc, ivc);
  wC = stagedWCenter(gc.wlo, iwc, hw);
  wHinv = Real(1) / (Real(0.5) * hw);
}

// face | Morton(iw, iu, iv): order of the parent elements in the upward packs
template <class Real>
IFGF_INLINE std::uint64_t packKey(const cones::FaceGrid<Real>& g, std::int32_t key)
{
  int f, iw, iu, iv;
  g.decode(key, f, iw, iu, iv);
  return (std::uint64_t(f) << 60) | spread3(iw) | spread3(iu) << 1 | spread3(iv) << 2;
}

template <class Real>
IFGF_INLINE void chebT(Real x, int n, Real* t)
{
  t[0] = Real(1), t[1] = x;
  for (int k = 2; k < n; ++k) t[k] = Real(2) * x * t[k - 1] - t[k - 2];
}

}  // namespace face_detail

// PO: angular interpolation order P, N0O: radial order (default P - 2); orders (N0O, P, P), 4 <= P <= 8, 2 <= N0O <= 8
template <class Kernel, int PO = 8, int N0O = (PO - 2 < 2 ? 2 : PO - 2)>
class FacePlan {
public:
  using Real = float;
  using Value = typename Kernel::Value;
  using SrcData = typename Kernel::SrcData;
  static constexpr int W = (std::is_floating_point_v<Value> ? 1 : 2) * Kernel::NC;
  static constexpr int G = 64;
  static constexpr int nearG = 128, nearCap = 256, nearR = std::is_empty_v<typename Kernel::SrcData> ? 2 : 4;
  static constexpr int FarR = 2;
  static constexpr int farChunkSize = G * FarR;
  static constexpr int farConeListCapacity = 512;
  static constexpr int farNSC = 3;                              // far field: cones staged per round
  static constexpr int leafCap = 128;
  static constexpr int UG = 256;  // upward work-group
  static_assert(PO >= 4 && PO <= 8 && N0O >= 2 && N0O <= 8, "FacePlan: 4 <= P <= 8, 2 <= radial order <= 8");
  static constexpr int NW = N0O, NU = PO, NB = NU * NU, NODES = NW * NB, CW = NODES * 2;
  static constexpr int farCS = (CW + 3) / 4 * 4 + 4;  // far field: staged cone stride (padded: other bank quad)
  // parent elements per upward work-group: values of all of them in local memory (AdaptiveCpp: <= 48 KB per group)
  static constexpr int UPCAP = NODES <= 384 ? 8 : 6;

  FacePlan(sycl::queue& q, std::span<const Vec3d> sources, std::span<const Vec3d> targets, std::complex<double> kappa,
           const FacePlanOptions& opt = {}, std::span<const Vec3d> sourceNormals = {})
      : q_(q), opt_(opt), orders_(interp::Orders{{N0O, PO, PO}})
  {
    static_assert(Kernel::NC == 1 && W == 2, "FacePlan: complex scalar kernels (W = 2)");
    if constexpr (!std::is_empty_v<SrcData>)
      if (sourceNormals.size() != sources.size()) throw std::invalid_argument("FacePlan: the kernel needs source normals");
    const std::size_t mem0 = exec::deviceBytesInUse();
    exec::resetDeviceBytesPeak();
    const auto t0 = std::chrono::steady_clock::now();
    auto lap = [t = t0]() mutable {
      const auto n = std::chrono::steady_clock::now();
      const double d = std::chrono::duration<double, std::milli>(n - t).count();
      t = n;
      return d;
    };
    S_ = buildTree<double>(sources, {opt.leafSize, 21});
    T_ = buildTree<double>(targets, {opt.leafSize, 21});
    stats_.treeMs = lap();
    lists_ = buildLists(S_, T_, {opt.alpha});
    stats_.listsMs = lap();
    kernel_ = Kernel(kappa, static_cast<Real>(opt.selfTolRel * std::max(S_.side, T_.side)));
    cones::FaceParams fp = opt.face;
    fp.alpha = opt.alpha;
    fp.density = opt.coneC * std::abs(kappa);  // elements per unit box side
    {
      auto r = cones::buildFaceConesDevice<Real>(q_, S_, T_, lists_, orders_, fp);
      cones_ = std::move(r.tables);
      devMaps_ = std::move(r.maps);
    }
    stats_.conesMs = lap();
    stats_.farActivationMs = cones_.farActivationMs;
    stats_.parentActivationMs = cones_.parentActivationMs;
    stats_.levelMapMs = cones_.levelMapMs;
    {
      double maxLeafSide = 0;
      for (std::int32_t b = 0; b < S_.numBoxes(); ++b)
        if (S_.isLeaf[b]) maxLeafSide = std::max(maxLeafSide, S_.boxSide(b));
      constexpr double pi = 3.14159265358979323846;
      leafPhaseSmall_ = std::abs(kappa.real()) * std::sqrt(3.0) / 2 * maxLeafSide <= 0.9 * 100 * pi;
    }
    stats_.nearPairs = lists_.near.numPairs();
    stats_.farPairs = lists_.far.numPairs();
    stats_.cones = cones_.numCones();
    for (std::int32_t b = 0; b < S_.numBoxes(); ++b) {
      const auto l = static_cast<std::size_t>(S_.level[b]);
      if (stats_.conesPerLevel.size() <= l) stats_.conesPerLevel.resize(l + 1, 0);
      stats_.conesPerLevel[l] += cones_.coneOffset[b + 1] - cones_.coneOffset[b];
    }
    const auto t1 = std::chrono::steady_clock::now();
    stats_.buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    upload(sourceNormals);
    stats_.uploadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    stats_.deviceBytes = exec::deviceBytesInUse() - mem0;
    stats_.initPeakDeviceBytes = exec::deviceBytesPeak() - mem0;
  }

  // Device memory held by the plan, by group ("other": the rest of stats().deviceBytes)
  std::vector<std::pair<const char*, std::size_t>> deviceMemory() const
  {
    auto b = [](const auto&... x) { return (std::size_t{0} + ... + x.bytes()); };
    std::size_t maps = 0;
    for (const auto& m : devMaps_)
      maps += b(m.U, m.uIdx, m.childOff, m.lineOff, m.child, m.task, m.line, m.lineNode);
    std::vector<std::pair<const char*, std::size_t>> g{
        {"coefficients", b(coef_)},
        {"upward plan", b(upGroups_, upElem_, upGSlot_, upStaged_, upTasks_, upLines_, upLineNodes_)},
        {"level maps", maps},
        {"cones", b(coneOff_, coneEl_, grids_, cheb_, tab_, sH_)},
        {"far lists", b(farPairs_, farOff_, farBox_, farBegin_, farCenterLo_)},
        {"near lists", b(nearPairs_, nearOff_, nearBox_)},
        {"points", b(srcPts_, tgtPts_, tgtLoc_, srcPtsLo_, tgtLocLo_, srcNrmLo_, srcRec_, srcNrm_, srcWi_, tgtPtsLo_)},
        {"boxes", b(sLevel_, sBegin_, sEnd_, tBegin_, tEnd_, leafBox_, leafSlot_)},
        {"weights, result", b(w_, result_, misses_)}};
    std::size_t sum = 0;
    for (const auto& [n, v] : g) sum += v;
    g.emplace_back("other", stats_.deviceBytes > sum ? stats_.deviceBytes - sum : 0);
    return g;
  }

  // New source normals (input order) for the next applies; trees, cones and plans stay. Only kernels with normals.
  void setSourceNormals(std::span<const Vec3d> normals)
  {
    if constexpr (std::is_empty_v<SrcData>) {
      throw std::logic_error("FacePlan::setSourceNormals: the kernel has no normals");
    } else {
      if (normals.size() != S_.numPoints()) throw std::invalid_argument("FacePlan::setSourceNormals: one normal per source");
      std::vector<sycl::vec<Real, 4>> hi(S_.numPoints()), lo(S_.numPoints());
      for (std::uint32_t i = 0; i < S_.numPoints(); ++i) {  // as upload(): tree order, hi/lo pairs of the doubles
        const Vec3d v = normals[S_.perm[i]];
        const Vec3f h = vec3Cast<Real>(v), l = vec3Cast<Real>(v - vec3Cast<double>(h));
        hi[i] = sycl::vec<Real, 4>{h.x, h.y, h.z, Real(0)};
        lo[i] = sycl::vec<Real, 4>{l.x, l.y, l.z, Real(0)};
      }
      srcNrm_.upload(hi);
      srcNrmLo_.upload(lo);
    }
  }

  std::vector<Value> apply(std::span<const Value> weights)
  {
    if (weights.size() != S_.numPoints()) throw std::invalid_argument("FacePlan::apply: one weight per source");
    const auto tStart = std::chrono::steady_clock::now();
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::vector<Value> ws(weights.size());
    for (std::size_t i = 0; i < ws.size(); ++i) ws[i] = weights[S_.perm[i]];
    w_.upload(ws);
    q_.fill(result_.data(), Value(0), result_.size());
    q_.fill(misses_.data(), 0, 1).wait();

    auto t = now();
    packSources();
    near();
    q_.wait_and_throw();
    stats_.nearMs = ms(t, now());
    t = now();
    if (numLeafWork_ > 0) leaf();
    q_.wait_and_throw();
    stats_.leafMs = ms(t, now());
    t = now();
    stats_.upwardLevelMs.assign(S_.numLevels(), 0.0);
    for (int l = S_.numLevels() - 2; l >= 0; --l) {
      if (upSlotLevelBegin_[l + 1] == upSlotLevelBegin_[l]) continue;
      const auto tl = now();
      upMain(l);  // structured tasks, fallback lines, transform
      q_.wait_and_throw();
      stats_.upwardLevelMs[l] = ms(tl, now());
    }
    stats_.upwardMs = ms(t, now());
    t = now();
    for (int l = 0; l < T_.numLevels(); ++l)
      if (farLevelBegin_[l + 1] > farLevelBegin_[l]) far(l);
    q_.wait_and_throw();
    stats_.farMs = ms(t, now());
    const auto rs = result_.download();
    stats_.lookupMisses = misses_.download()[0];
    std::vector<Value> out(rs.size());
    for (std::size_t i = 0; i < rs.size(); ++i) out[T_.perm[i]] = rs[i];
    stats_.applyMs = ms(tStart, now());
    return out;
  }

  const FacePlanStats& stats() const { return stats_; }
  const Tree& sourceTree() const { return S_; }
  const Tree& targetTree() const { return T_; }

private:
  void upload(std::span<const Vec3d> normals)
  {
    using exec::DeviceBuffer;
    using namespace face_detail;
    const Vec3d c0 = cones_.c0;
    auto leafCenters = [](const Tree& tr) {
      std::vector<Vec3d> c(tr.numPoints());
      for (std::int32_t b = 0; b < tr.numBoxes(); ++b)
        if (tr.isLeaf[b])
          for (auto i = tr.ptBegin[b]; i < tr.ptEnd[b]; ++i) c[i] = tr.center[b];
      return c;
    };
    const auto sLeafC = leafCenters(S_), tLeafC = leafCenters(T_);
    // leaf-local positions (and normals) as hi/lo float pairs of the double values: the near field forms s - t and
    // (s - t).n from them without losing the O(r^2) part (compensated near field)
    auto hiLo = [](const Vec3d& v, Vec3f& h, Vec3f& l) {
      h = vec3Cast<Real>(v);
      l = vec3Cast<Real>(v - vec3Cast<double>(h));
    };
    std::vector<Vec3f> sp(S_.numPoints()), spLo(S_.numPoints()), tLoc(T_.numPoints()), tLocLo(T_.numPoints()), tRel(T_.numPoints()),
        tRelLo(T_.numPoints());
    std::vector<SrcData> sd(S_.numPoints());
    std::vector<sycl::vec<Real, 4>> nLo;
    for (std::uint32_t i = 0; i < S_.numPoints(); ++i) {
      hiLo(S_.points[i] - sLeafC[i], sp[i], spLo[i]);
      if constexpr (!std::is_empty_v<SrcData>) {
        Vec3f h, l;
        hiLo(normals[S_.perm[i]], h, l);
        sd[i].n = h;
        nLo.push_back(sycl::vec<Real, 4>{l.x, l.y, l.z, Real(0)});
      }
    }
    for (std::uint32_t i = 0; i < T_.numPoints(); ++i) {
      hiLo(T_.points[i] - tLeafC[i], tLoc[i], tLocLo[i]);
      hiLo(T_.points[i] - c0, tRel[i], tRelLo[i]);
    }
    srcPts_ = DeviceBuffer<Vec3f>(q_, sp);
    srcPtsLo_ = DeviceBuffer<Vec3f>(q_, spLo);
    tgtPts_ = DeviceBuffer<Vec3f>(q_, tRel);
    tgtLoc_ = DeviceBuffer<Vec3f>(q_, tLoc);
    tgtLocLo_ = DeviceBuffer<Vec3f>(q_, tLocLo);
    tgtPtsLo_ = DeviceBuffer<Vec3f>(q_, tRelLo);  // compensated far-field phase (tgtPts_ is the hi part)
    {
      std::vector<Vec3f> cl(lists_.far.src.size());
      for (std::size_t n = 0; n < cl.size(); ++n) {
        Vec3f h;
        hiLo(S_.center[lists_.far.src[n]] - c0, h, cl[n]);  // h = FarPairF::center
      }
      farCenterLo_ = DeviceBuffer<Vec3f>(q_, cl);
    }
    srcNrmLo_ = nLo.empty() ? DeviceBuffer<sycl::vec<Real, 4>>(q_, 1) : DeviceBuffer<sycl::vec<Real, 4>>(q_, nLo);
    std::vector<std::int32_t> sl(S_.numBoxes());
    for (std::int32_t b = 0; b < S_.numBoxes(); ++b) sl[b] = S_.level[b];
    std::vector<Real> sH(S_.H.begin(), S_.H.end());
    sLevel_ = DeviceBuffer<std::int32_t>(q_, sl);
    sBegin_ = DeviceBuffer<std::uint32_t>(q_, S_.ptBegin);
    sEnd_ = DeviceBuffer<std::uint32_t>(q_, S_.ptEnd);
    sH_ = DeviceBuffer<Real>(q_, sH);
    tBegin_ = DeviceBuffer<std::uint32_t>(q_, T_.ptBegin);
    tEnd_ = DeviceBuffer<std::uint32_t>(q_, T_.ptEnd);
    // near field
    nearOff_ = DeviceBuffer<std::int32_t>(q_, lists_.near.offsets);
    {
      std::vector<NearPairC> np(lists_.near.src.size());
      for (std::int32_t t = 0; t < T_.numBoxes(); ++t) {
        std::uint32_t cum = 0;
        for (auto k = lists_.near.offsets[t]; k < lists_.near.offsets[t + 1]; ++k) {
          const std::int32_t sb = lists_.near.src[k];
          hiLo(S_.center[sb] - T_.center[t], np[k].d, np[k].dl);
          np[k].s0 = S_.ptBegin[sb];
          np[k].n = S_.numPoints(sb);
          np[k].cum = cum;
          cum += np[k].n;
        }
      }
      nearPairs_ = DeviceBuffer<NearPairC>(q_, np);
      srcRec_ = DeviceBuffer<sycl::vec<Real, 4>>(q_, S_.numPoints());
      srcWi_ = DeviceBuffer<Real>(q_, S_.numPoints());
      std::vector<sycl::vec<Real, 4>> nn;
      if constexpr (!std::is_empty_v<SrcData>)
        for (const auto& d : sd) nn.push_back(sycl::vec<Real, 4>{d.n.x, d.n.y, d.n.z, Real(0)});
      srcNrm_ = nn.empty() ? DeviceBuffer<sycl::vec<Real, 4>>(q_, 1) : DeviceBuffer<sycl::vec<Real, 4>>(q_, nn);
    }
    farOff_ = DeviceBuffer<std::int32_t>(q_, lists_.far.offsets);
    {
      std::vector<FarPairF<Real>> fpv(lists_.far.src.size());
      for (std::size_t n = 0; n < fpv.size(); ++n) {
        const std::int32_t s = lists_.far.src[n];
        fpv[n].grid = cones_.grid[S_.level[s]];
        fpv[n].center = vec3Cast<Real>(S_.center[s] - c0);
        fpv[n].H = static_cast<Real>(S_.boxSide(s));
        fpv[n].coneBegin = cones_.coneOffset[s];
        fpv[n].coneEnd = cones_.coneOffset[s + 1];
      }
      farPairs_ = DeviceBuffer<FarPairF<Real>>(q_, fpv);
    }
    grids_ = DeviceBuffer<cones::FaceGrid<Real>>(q_, cones_.grid);
    coneOff_ = DeviceBuffer<std::int32_t>(q_, cones_.coneOffset);
    coneEl_ = DeviceBuffer<std::int32_t>(q_, cones_.coneEl);
    std::vector<Real> cheb;
    for (int d = 0; d < 3; ++d) {
      chebOff_[d] = static_cast<int>(cheb.size());
      for (int i = 0; i < orders_.n[d]; ++i) cheb.push_back(static_cast<Real>(interp::chebNode(i, orders_.n[d])));
    }
    cheb_ = DeviceBuffer<Real>(q_, cheb);
    const interp::TransformTables<Real> tab(orders_);
    tab_ = DeviceBuffer<Real>(q_, tab.data);
    for (int d = 0; d < 3; ++d) dtab_.M[d] = tab_.data() + tab.offset(d);

    // leaf work, parent slots per level (transform), far chunks, near boxes
    std::vector<std::int32_t> leafBox, leafSlot, upSlot, farBox, farBegin, nearBox;
    upSlotLevelBegin_.assign(S_.numLevels() + 1, 0);
    for (int l = 0; l < S_.numLevels(); ++l) {
      upSlotLevelBegin_[l] = static_cast<std::int32_t>(upSlot.size());
      for (auto b = S_.levelBegin[l]; b < S_.levelBegin[l + 1]; ++b)
        for (auto k = cones_.coneOffset[b]; k < cones_.coneOffset[b + 1]; ++k) {
          if (S_.isLeaf[b]) {
            leafBox.push_back(b);
            leafSlot.push_back(k);
          } else {
            upSlot.push_back(k);
          }
        }
    }
    upSlotLevelBegin_[S_.numLevels()] = static_cast<std::int32_t>(upSlot.size());
    farLevelBegin_.assign(T_.numLevels() + 1, 0);
    for (int l = 0; l < T_.numLevels(); ++l) {
      farLevelBegin_[l] = static_cast<std::int32_t>(farBox.size());
      for (auto t = T_.levelBegin[l]; t < T_.levelBegin[l + 1]; ++t)
        if (lists_.far.offsets[t + 1] > lists_.far.offsets[t])
          for (auto b = T_.ptBegin[t]; b < T_.ptEnd[t]; b += farChunkSize) {
            farBox.push_back(t);
            farBegin.push_back(static_cast<std::int32_t>(b));
          }
    }
    farLevelBegin_[T_.numLevels()] = static_cast<std::int32_t>(farBox.size());
    for (std::int32_t t = 0; t < T_.numBoxes(); ++t)
      if (lists_.near.offsets[t + 1] > lists_.near.offsets[t]) nearBox.push_back(t);
    numLeafWork_ = leafBox.size();
    numNearWork_ = nearBox.size();
    leafBox_ = DeviceBuffer<std::int32_t>(q_, leafBox);
    leafSlot_ = DeviceBuffer<std::int32_t>(q_, leafSlot);
    farBox_ = DeviceBuffer<std::int32_t>(q_, farBox);
    farBegin_ = DeviceBuffer<std::int32_t>(q_, farBegin);
    nearBox_ = DeviceBuffer<std::int32_t>(q_, nearBox);

    const auto tUp = std::chrono::steady_clock::now();
    buildUpwardDevice();
    stats_.upPlanMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tUp).count();

    coef_ = DeviceBuffer<Real>(q_, static_cast<std::size_t>(cones_.numCones()) * orders_.nodes() * W);
    w_ = DeviceBuffer<Value>(q_, S_.numPoints());
    result_ = DeviceBuffer<Value>(q_, T_.numPoints());
    misses_ = DeviceBuffer<int>(q_, 1);
  }

  // Upward plan, built on the device:
  //   1. pack keys (face | Morton) per cone slot of the non-leaf parent boxes; rank sort per box (O(n^2) per box,
  //      one work-group per box) -> slots in pack order
  //   2. packs per box (one work item per box: runs of one face, at most UPCAP elements): count, host scan, fill
  //   3. per (pack, octant): merge the child keys of the pack's elements (level map), count staged child cones and
  //      structured + line tasks; host scan; fill straight into the final device buffers
  // The box-independent level maps are read on the device (devMaps_).
  void buildUpwardDevice()
  {
    using namespace face_detail;
    using exec::DeviceBuffer;
    constexpr int GB = 256;
    const int L = S_.numLevels();
    upGroupLevelBegin_.assign(L + 1, 0);
    lineLevelBegin_.assign(L + 1, 0);
    // fallback lines and their nodes (box-independent): copied per level from the device level maps into one buffer
    // each, the level's line node offset added on the device
    std::size_t numLines = 0, numLineNodes = 0;
    {
      std::vector<std::size_t> nodeBegin(L + 1, 0);
      for (int l = 0; l < L; ++l) {
        lineLevelBegin_[l] = static_cast<std::int32_t>(numLines);
        nodeBegin[l] = numLineNodes;
        if (l + 1 >= L) continue;
        const auto& D = devMaps_[l];
        if (D.numLines >= (std::size_t(1) << lineShift)) throw std::runtime_error("FacePlan: too many fallback lines");
        numLines += D.numLines, numLineNodes += D.numLineNodes;
      }
      lineLevelBegin_[L] = static_cast<std::int32_t>(numLines);
      if (numLineNodes >= std::size_t(std::numeric_limits<std::int32_t>::max()))
        throw std::runtime_error("FacePlan: fallback line nodes overflow int32");
      upLines_ = DeviceBuffer<cones::FaceLine>(q_, std::max<std::size_t>(numLines, 1));
      upLineNodes_ = DeviceBuffer<cones::FaceLineNode>(q_, std::max<std::size_t>(numLineNodes, 1));
      if (numLines == 0) q_.fill(upLines_.data(), cones::FaceLine{}, 1);
      if (numLineNodes == 0) q_.fill(upLineNodes_.data(), cones::FaceLineNode{}, 1);
      for (int l = 0; l + 1 < L; ++l) {
        const auto& D = devMaps_[l];
        if (D.numLines == 0) continue;
        cones::FaceLine* dl = upLines_.data() + lineLevelBegin_[l];
        q_.copy(D.line.data(), dl, D.numLines);
        if (D.numLineNodes) q_.copy(D.lineNode.data(), upLineNodes_.data() + nodeBegin[l], D.numLineNodes);
        const std::int32_t nb = static_cast<std::int32_t>(nodeBegin[l]);
        if (nb != 0) q_.parallel_for(sycl::range<1>{D.numLines}, [=](sycl::id<1> i) { dl[i[0]].nBegin += nb; });
      }
      q_.wait_and_throw();
    }

    // parent boxes in host order (level, box id): non-leaf with cones, children on a level < L
    std::vector<std::int32_t> pbs;
    std::vector<std::int32_t> pbLevelBegin(L + 1, 0);
    for (int l = 0; l < L; ++l) {
      pbLevelBegin[l] = static_cast<std::int32_t>(pbs.size());
      if (l + 1 >= L) continue;
      for (auto pb = S_.levelBegin[l]; pb < S_.levelBegin[l + 1]; ++pb)
        if (!S_.isLeaf[pb] && cones_.coneOffset[pb + 1] > cones_.coneOffset[pb]) pbs.push_back(pb);
    }
    pbLevelBegin[L] = static_cast<std::int32_t>(pbs.size());
    const std::size_t NP = pbs.size();
    std::vector<UpGroup> groupsH;
    std::size_t numStaged = 0, numTasks = 0;
    if (NP > 0) {
      DeviceBuffer<std::int32_t> dPbs(q_, pbs);
      const std::int32_t *pp = dPbs.data(), *co = coneOff_.data(), *ce = coneEl_.data(), *lv = sLevel_.data();
      const cones::FaceGrid<Real>* gr = grids_.data();
      // 1. pack keys and rank sort per box -> sorted[slot range of the box] = slots in pack order
      const std::size_t nC = static_cast<std::size_t>(cones_.numCones());
      DeviceBuffer<std::uint64_t> mk(q_, nC);
      DeviceBuffer<std::int32_t> sorted(q_, nC);
      std::uint64_t* mkp = mk.data();
      std::int32_t* sp = sorted.data();
      q_.parallel_for(sycl::nd_range<1>{NP * GB, GB}, [=](sycl::nd_item<1> it) {
         const std::int32_t pb = pp[it.get_group(0)];
         const cones::FaceGrid<Real> g = gr[lv[pb]];
         for (std::int32_t s = co[pb] + static_cast<std::int32_t>(it.get_local_id(0)); s < co[pb + 1]; s += GB)
           mkp[s] = packKey(g, ce[s]);
       });
      q_.parallel_for(sycl::nd_range<1>{NP * GB, GB}, [=](sycl::nd_item<1> it) {
         const std::int32_t pb = pp[it.get_group(0)], c0 = co[pb], c1 = co[pb + 1];
         for (std::int32_t s = c0 + static_cast<std::int32_t>(it.get_local_id(0)); s < c1; s += GB) {
           const std::uint64_t k = mkp[s];
           std::int32_t r = 0;
           for (std::int32_t j = c0; j < c1; ++j) r += mkp[j] < k ? 1 : 0;
           sp[c0 + r] = s;
         }
       }).wait_and_throw();
      // 2. packs per box: count, scan, fill
      DeviceBuffer<std::int32_t> gcnt(q_, NP);
      std::int32_t* gcp = gcnt.data();
      auto packs = [=](std::int32_t pb, auto&& emit) {
        const std::int32_t c0 = co[pb], c1 = co[pb + 1];
        for (std::int32_t a = c0; a < c1;) {
          std::int32_t b = a;
          while (b < c1 && b - a < UPCAP && (mkp[sp[b]] >> 60) == (mkp[sp[a]] >> 60)) ++b;
          emit(a, b);
          a = b;
        }
      };
      q_.parallel_for(sycl::range<1>{NP}, [=](sycl::id<1> id) {
         std::int32_t n = 0;
         packs(pp[id[0]], [&](std::int32_t, std::int32_t) { ++n; });
         gcp[id[0]] = n;
       }).wait_and_throw();
      const auto gc = gcnt.download();
      std::vector<std::int32_t> gOff(NP + 1, 0);
      for (std::size_t p = 0; p < NP; ++p) gOff[p + 1] = gOff[p] + gc[p];
      for (int l = 0; l <= L; ++l) upGroupLevelBegin_[l] = gOff[pbLevelBegin[l]];
      const std::size_t NG = static_cast<std::size_t>(gOff[NP]);
      DeviceBuffer<std::int32_t> dGOff(q_, gOff);
      upElem_ = DeviceBuffer<std::int32_t>(q_, NG * UPCAP);
      upGSlot_ = DeviceBuffer<std::int32_t>(q_, NG * UPCAP);
      DeviceBuffer<std::int32_t> gFace(q_, NG), gN(q_, NG), gBox(q_, NG);
      {
        const std::int32_t* go = dGOff.data();
        std::int32_t *el = upElem_.data(), *sl = upGSlot_.data(), *gf = gFace.data(), *gn = gN.data(), *gb = gBox.data();
        q_.parallel_for(sycl::range<1>{NP}, [=](sycl::id<1> id) {
           const std::int32_t pb = pp[id[0]];
           const cones::FaceGrid<Real> g = gr[lv[pb]];
           std::int32_t q = go[id[0]];
           packs(pb, [&](std::int32_t a, std::int32_t b) {
             gf[q] = static_cast<std::int32_t>(mkp[sp[a]] >> 60);
             gn[q] = b - a;
             gb[q] = pb;
             for (int e = 0; e < UPCAP; ++e) {
               if (a + e < b) {
                 const std::int32_t s = sp[a + e];
                 int f, iw, iu, iv;
                 g.decode(ce[s], f, iw, iu, iv);
                 el[q * UPCAP + e] = iw | iu << 10 | iv << 20;
                 sl[q * UPCAP + e] = s;
               } else {
                 el[q * UPCAP + e] = -1;
                 sl[q * UPCAP + e] = -1;
               }
             }
             ++q;
           });
         }).wait_and_throw();
      }
      // 3. per (pack, octant): count, scan, fill; level maps read on the device (cones builder)
      std::vector<std::int32_t> childFlat(static_cast<std::size_t>(S_.numBoxes()) * 8);
      for (std::int32_t b = 0; b < S_.numBoxes(); ++b)
        for (int o = 0; o < 8; ++o) childFlat[b * 8 + o] = S_.child[b][o];
      DeviceBuffer<std::int32_t> dChild(q_, childFlat);
      std::vector<DeviceBuffer<Real>> hwCol(L);  // per child column (iu, iv): hw, 1 / (hw / 2) (host values)
      DeviceBuffer<std::int32_t> cnt(q_, NG * 8 * 3);  // per item: staged, structured tasks, line tasks
      q_.fill(cnt.data(), 0, cnt.size()).wait();
      // the item kernel: Fill = false counts, Fill = true writes
      auto items = [&](int l, auto fillC, const std::int64_t* stOff, const std::int64_t* tkOff, UpStaged* stOut,
                       std::uint32_t* tkOut, int* miss) {
        constexpr bool Fill = decltype(fillC)::value;
        const std::size_t g0 = upGroupLevelBegin_[l], g1 = upGroupLevelBegin_[l + 1];
        if (g1 == g0) return;
        const auto& D = devMaps_[l];
        const std::int32_t *ui = D.uIdx.data(), *cho = D.childOff.data(), *lno = D.lineOff.data(), *chb = dChild.data(),
                           *gsl = upGSlot_.data(), *gn = gN.data(), *gb = gBox.data();
        const cones::FaceLine* lnr = D.line.data();
        const cones::FaceChildRec* chr = D.child.data();
        const std::uint32_t* tsk = D.task.data();
        const cones::FaceGrid<Real> gch = cones_.grid[l + 1];
        const Real* hwc = hwCol[l].data();
        std::int32_t* cp = cnt.data();
        q_.parallel_for(sycl::range<1>{(g1 - g0) * 8}, [=](sycl::id<1> id) {
           const std::size_t g = g0 + id[0] / 8, item = g * 8 + id[0] % 8;
           const int oct = static_cast<int>(id[0] % 8);
           const std::int32_t cb = chb[gb[g] * 8 + oct];
           if (cb < 0) return;
           const int n = gn[g];
           std::int32_t cB[UPCAP], cE[UPCAP], lB[UPCAP], lE[UPCAP];
           for (int e = 0; e < UPCAP; ++e) {
             cB[e] = cE[e] = lB[e] = lE[e] = 0;
             if (e < n) {
               const std::int32_t idx = ui[ce[gsl[g * UPCAP + e]]] * 8 + oct;
               cB[e] = cho[idx], cE[e] = cho[idx + 1], lB[e] = lno[idx], lE[e] = lno[idx + 1];
             }
           }
           std::int64_t st = Fill ? stOff[item] : 0, tk = Fill ? tkOff[item] : 0;
           std::int32_t nSt = 0, nTk = 0, nLn = 0;
           constexpr std::int32_t none = 0x7fffffff;
           for (;;) {
             std::int32_t key = none;
             for (int e = 0; e < n; ++e) {
               if (cB[e] < cE[e] && chr[cB[e]].key < key) key = chr[cB[e]].key;
               if (lB[e] < lE[e] && lnr[lB[e]].key < key) key = lnr[lB[e]].key;
             }
             if (key == none) break;
             const std::int64_t tBegin = tk;
             for (int e = 0; e < n; ++e)
               if (cB[e] < cE[e] && chr[cB[e]].key == key) {
                 const cones::FaceChildRec r = chr[cB[e]];
                 for (auto t = r.taskBegin; t < r.taskEnd; ++t) {
                   if constexpr (Fill) tkOut[tk] = static_cast<std::uint32_t>(e) << 20 | tsk[t];
                   ++tk, ++nTk;
                 }
                 ++cB[e];
               }
             const std::int64_t lBegin = tk;
             for (int e = 0; e < n; ++e)
               for (; lB[e] < lE[e] && lnr[lB[e]].key == key; ++lB[e]) {
                 if constexpr (Fill) tkOut[tk] = static_cast<std::uint32_t>(e) << lineShift | static_cast<std::uint32_t>(lB[e]);
                 ++tk, ++nLn;
               }
             if constexpr (Fill) {
               // slot of the child cone: binary search in the child box's sorted cone list
               std::int32_t lo = co[cb], hi = co[cb + 1];
               while (lo < hi) {
                 const std::int32_t mid = (lo + hi) / 2;
                 if (ce[mid] < key) lo = mid + 1;
                 else hi = mid;
               }
               std::int32_t slot = lo;
               if (lo >= co[cb + 1] || ce[lo] != key) {
                 slot = -1;
                 sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>(*miss)
                     .fetch_add(1);
               }
               int f, iwc, iuc, ivc;
               gch.decode(key, f, iwc, iuc, ivc);
               UpStaged s{oct, slot, iuc, ivc, static_cast<std::int32_t>(tBegin), static_cast<std::int32_t>(lBegin),
                          0.f, 0.f, static_cast<std::int32_t>(lBegin), static_cast<std::int32_t>(tk), 0, 0};
               const std::int32_t col = 2 * (iuc * gch.span + ivc);
               s.wC = stagedWCenter(gch.wlo, iwc, hwc[col]);
               s.wHinv = hwc[col + 1];
               stOut[st] = s;
               ++st;
             }
             ++nSt;
           }
           if constexpr (!Fill) {
             cp[item * 3] = nSt;
             cp[item * 3 + 1] = nTk;
             cp[item * 3 + 2] = nLn;
           }
         }).wait_and_throw();
      };
      for (int l = 0; l + 1 < L; ++l) {
        if (upGroupLevelBegin_[l + 1] == upGroupLevelBegin_[l]) continue;
        {
          const auto& gc = cones_.grid[l + 1];
          std::vector<Real> hwv(static_cast<std::size_t>(2) * gc.span * gc.span);
          for (int iu = 0; iu < gc.span; ++iu)
            for (int iv = 0; iv < gc.span; ++iv) {
              float wc, wh;
              stagedW(gc, 0, iu, iv, wc, wh);
              hwv[2 * (iu * gc.span + iv)] = gc.hw(iu, iv);
              hwv[2 * (iu * gc.span + iv) + 1] = wh;
            }
          hwCol[l] = DeviceBuffer<Real>(q_, hwv);
        }
        items(l, std::false_type{}, nullptr, nullptr, nullptr, nullptr, nullptr);
      }
      const auto cv = cnt.download();
      std::vector<std::int64_t> stOff(NG * 8 + 1, 0), tkOff(NG * 8 + 1, 0);
      for (std::size_t i = 0; i < NG * 8; ++i) {
        stOff[i + 1] = stOff[i] + cv[i * 3];
        tkOff[i + 1] = tkOff[i] + cv[i * 3 + 1] + cv[i * 3 + 2];
      }
      numStaged = static_cast<std::size_t>(stOff[NG * 8]);
      numTasks = static_cast<std::size_t>(tkOff[NG * 8]);
      if (numTasks >= std::size_t(std::numeric_limits<std::int32_t>::max()))
        throw std::runtime_error("FacePlan: upward tasks overflow int32");
      upStaged_ = DeviceBuffer<UpStaged>(q_, std::max<std::size_t>(numStaged, 1));
      upTasks_ = DeviceBuffer<std::uint32_t>(q_, std::max<std::size_t>(numTasks, 1));
      DeviceBuffer<std::int64_t> dSt(q_, stOff), dTk(q_, tkOff);
      DeviceBuffer<int> miss(q_, 1);
      q_.fill(miss.data(), 0, 1).wait();
      for (int l = 0; l + 1 < L; ++l)
        items(l, std::true_type{}, dSt.data(), dTk.data(), upStaged_.data(), upTasks_.data(), miss.data());
      if (miss.download()[0] != 0) throw std::logic_error("FacePlan: a child cone of the upward plan is not active");
      // groups: (face, n, staged range)
      const auto gf = gFace.download(), gnH = gN.download();
      groupsH.resize(NG);
      for (std::size_t g = 0; g < NG; ++g)
        groupsH[g] = {gf[g], gnH[g], static_cast<std::int32_t>(stOff[g * 8]), static_cast<std::int32_t>(stOff[g * 8 + 8])};
    }
    if (groupsH.empty()) {
      groupsH.push_back({0, 0, 0, 0});
      upElem_ = DeviceBuffer<std::int32_t>(q_, std::vector<std::int32_t>(UPCAP, -1));
      upGSlot_ = DeviceBuffer<std::int32_t>(q_, std::vector<std::int32_t>(UPCAP, -1));
    }
    if (numStaged == 0) upStaged_ = DeviceBuffer<UpStaged>(q_, std::vector<UpStaged>{UpStaged{}});
    if (numTasks == 0) upTasks_ = DeviceBuffer<std::uint32_t>(q_, std::vector<std::uint32_t>{0});
    upGroups_ = DeviceBuffer<UpGroup>(q_, groupsH);
  }

  void packSources()
  {
    constexpr Real inv4pi = kernels::detail::inv4pi<Real>;
    using V4 = sycl::vec<Real, 4>;
    const Vec3f* sp = srcPts_.data();
    const Value* w = w_.data();
    V4* rec = srcRec_.data();
    Real* rwi = srcWi_.data();
    q_.parallel_for(sycl::range<1>{S_.numPoints()}, [=](sycl::id<1> id) {
      const std::size_t i = id[0];
      const Vec3f s = sp[i];
      const Value wv = w[i];
      rec[i] = V4{s.x, s.y, s.z, wv.real() * inv4pi};
      rwi[i] = wv.imag() * inv4pi;
    });
  }

  // near field: group per target box with near pairs, direct sum with the geometry in compensated FP32: source and
  // target positions (leaf-local), the box offset and the normals are hi/lo pairs of the doubles. For close points
  // on a curved surface (s - t).n ~ r^2 / 2R is O(r^2) while its factors are O(r), so plain FP32 would lose it. Here
  //   (s - t).n = sigma - tau,  sigma = s.n once per staged source (dot2, amortised over the group's targets),
  //   tau = t.n per interaction (dot2), the difference of the two hi/lo pairs (the hi difference is exact when they
  //   cancel: Sterbenz);
  //   s - t = fl(s_hi - t_hi) + (s_lo - t_lo): one rounding relative to |s - t| itself, enough for r^2.
  void near()
  {
    if (numNearWork_ == 0) return;
    using namespace face_detail;
    constexpr bool hasSd = !std::is_empty_v<SrcData>;
    constexpr int GN = nearG, CAP = nearCap, WS = 32, NWp = GN / WS, NR = nearR;
    using V4 = sycl::vec<Real, 4>;
    const Kernel k = kernel_;
    const std::int32_t *box = nearBox_.data(), *off = nearOff_.data();
    const NearPairC* pairs = nearPairs_.data();
    const std::uint32_t *tb = tBegin_.data(), *te = tEnd_.data();
    const Vec3f *tl = tgtLoc_.data(), *tll = tgtLocLo_.data(), *spl = srcPtsLo_.data();
    Value* res = result_.data();
    const V4 *rec = srcRec_.data(), *nrm = srcNrm_.data(), *nrml = srcNrmLo_.data();
    const Real* rwi = srcWi_.data();
    q_.submit([&](sycl::handler& h) {
      sycl::local_accessor<V4, 1> lp{sycl::range<1>{CAP}, h};   // hi of (s - c_t), weight
      sycl::local_accessor<V4, 1> llo{sycl::range<1>{CAP}, h};  // lo of (s - c_t)
      sycl::local_accessor<Real, 1> lwi{sycl::range<1>{CAP}, h};
      sycl::local_accessor<V4, 1> ln{sycl::range<1>{hasSd ? CAP : 1}, h};
      sycl::local_accessor<V4, 1> lnl{sycl::range<1>{hasSd ? CAP : 1}, h};
      sycl::local_accessor<Value, 1> lred{sycl::range<1>{GN}, h};
      h.parallel_for(sycl::nd_range<1>{numNearWork_ * GN, GN}, [=](sycl::nd_item<1> it) {
        const auto grp = it.get_group();
        const int lid = static_cast<int>(it.get_local_id(0)), wid = lid / WS, lane = lid % WS;
        const std::int32_t t = box[it.get_group(0)], p0 = off[t], p1 = off[t + 1];
        const std::uint32_t t0 = tb[t], nt = te[t] - t0;
        const std::uint32_t total = pairs[p1 - 1].cum + pairs[p1 - 1].n;
        const std::uint32_t numWin = (total + CAP - 1) / CAP;
        for (std::uint32_t a0 = 0; a0 < nt; a0 += NR * GN) {
          const std::uint32_t left = nt - a0;
          const int rows = left < NR * GN ? static_cast<int>((left + NR - 1) / NR) : GN, splits = GN / rows;
          const int a = lid % rows, b = lid / rows;
          const bool act = b < splits;
          Vec3f x[NR], xl[NR];
          for (int q = 0; q < NR; ++q) {
            const std::uint32_t ia = static_cast<std::uint32_t>(a + q * rows);
            const std::uint32_t ti = t0 + a0 + (ia < left ? ia : static_cast<std::uint32_t>(a));
            x[q] = tl[ti], xl[q] = tll[ti];
          }
          Real ar[NR] = {}, ai[NR] = {};
          std::int32_t pw = p0 + wid;
          auto stage = [&](std::uint32_t base) {
            const std::uint32_t end = total - base < CAP ? total : base + CAP;
            for (; pw < p1; pw += NWp) {
              const NearPairC P = pairs[pw];
              if (P.cum >= end) break;
              const std::uint32_t lo = P.cum > base ? P.cum : base, pe = P.cum + P.n, hi = pe < end ? pe : end;
              for (std::uint32_t i = lo + lane; i < hi; i += WS) {
                const std::uint32_t gI = P.s0 + (i - P.cum);
                const int l = static_cast<int>(i - base);
                const V4 r = rec[gI];
                const Vec3f sl = spl[gI];
                // (s - c_s) + (c_s - c_t) as hi/lo: hi = fl(s_hi + d), lo = rounding error + s_lo + d_lo
                Vec3f sh, se;
                comp::twoSum(r.x(), P.d.x, sh.x, se.x);
                comp::twoSum(r.y(), P.d.y, sh.y, se.y);
                comp::twoSum(r.z(), P.d.z, sh.z, se.z);
                const Vec3f lo{se.x + (sl.x + P.dl.x), se.y + (sl.y + P.dl.y), se.z + (sl.z + P.dl.z)};
                lp[l] = V4{sh.x, sh.y, sh.z, r.w()};
                lwi[l] = rwi[gI];
                Real sgh = Real(0), sgl = Real(0);
                if constexpr (hasSd) {
                  const V4 N = nrm[gI], NL = nrml[gI];
                  comp::dot2pair(sh, lo, Vec3f{N.x(), N.y(), N.z()}, Vec3f{NL.x(), NL.y(), NL.z()}, sgh, sgl);
                  ln[l] = V4{N.x(), N.y(), N.z(), sgl};
                  lnl[l] = NL;
                }
                llo[l] = V4{lo.x, lo.y, lo.z, sgh};  // sigma = s.n (hi in llo.w, lo in ln.w)
              }
              if (pe > end) break;
            }
          };
          for (std::uint32_t win = 0; win < numWin; ++win) {
            stage(win * CAP);
            sycl::group_barrier(grp);
            if (act) {
              const std::uint32_t base = win * CAP, cnt = total - base < CAP ? total - base : CAP;
              const std::uint32_t len = (cnt + static_cast<std::uint32_t>(splits) - 1) / static_cast<std::uint32_t>(splits);
              const std::uint32_t mb = static_cast<std::uint32_t>(b) * len, me = mb + len < cnt ? mb + len : cnt;
              const int mBegin = static_cast<int>(mb), mEnd = static_cast<int>(mb < cnt ? me : cnt);
#pragma unroll 2
              for (int m = mBegin; m < mEnd; ++m) {
                const V4 P = lp[m], PL = llo[m];
                const Vec3f ph{P.x(), P.y(), P.z()}, pl{PL.x(), PL.y(), PL.z()};
                Vec3f nh{}, nl{};
                Real sgl = Real(0);
                if constexpr (hasSd) {
                  const V4 N = ln[m], NL = lnl[m];
                  nh = Vec3f{N.x(), N.y(), N.z()}, nl = Vec3f{NL.x(), NL.y(), NL.z()}, sgl = N.w();
                }
                const Real wi = lwi[m];
#pragma unroll
                for (int q = 0; q < NR; ++q) {
                  const Vec3f dh = ph - x[q], dl = pl - xl[q];  // s - t
                  const Real r2 = comp::norm2c(dh, dl);
                  Real xn = Real(0);
                  if constexpr (hasSd) {
                    Real th, tl2;
                    comp::dot2pair(x[q], xl[q], nh, nl, th, tl2);  // tau = t.n
                    xn = (PL.w() - th) + (sgl - tl2);
                  }
                  const Value g = k.direct4piFastRX(r2, xn);
                  ar[q] += P.w() * g.real();
                  ar[q] -= wi * g.imag();
                  ai[q] += P.w() * g.imag();
                  ai[q] += wi * g.real();
                }
              }
            }
            sycl::group_barrier(grp);
          }
          for (int q = 0; q < NR; ++q) {
            Value acc(ar[q], ai[q]);
            lred[lid] = acc;
            sycl::group_barrier(grp);
            const std::uint32_t ia = static_cast<std::uint32_t>(a + q * rows);
            if (act && b == 0 && ia < left) {
              for (int qq = 1; qq < splits; ++qq) acc += lred[a + qq * rows];
              res[t0 + a0 + ia] += acc;
            }
            sycl::group_barrier(grp);
          }
        }
      });
    });
  }

  void leaf()
  {
    if (leafPhaseSmall_)
      leafG(std::false_type{});
    else
      leafG(std::true_type{});
  }

  // Leaf stage: node values of each (leaf box, active cone) = sum over the box's sources of w * factored kernel
  // at the face-chart nodes, then the transform -> coefficients. One thread per ray (u-node j, v-node k): its NW
  // radial nodes t = r u share the direction u, so per (ray, source) p = u.s once, and per node A = |s|^2 - 2 r p
  // (the exact phase numerator), d^2 = r^2 + A. delta = d - r without a reciprocal: delta0 = (d + r) - 2r, then one
  // FMA residual A - delta0 (d + r) scaled by 1 / 2r removes the cancellation error. 3 special-function operations
  // per pair (rsqrt, sin, cos); no self guard (the nodes lie outside the box).
  // Kernels with source normals (DL, CF): (s - t).n = s.n - r (u.n): s.n staged per source, u.n per (ray, source),
  // per node one FMA, then the kernel's rayFactor(1/d, (s - t).n / d). t lies outside the box: plain FP32.
  // Reduce: range-reduced sin/cos (needed when the leaf boxes are large compared with the wavelength).
  template <class ReduceT>
  void leafG(ReduceT)
  {
    constexpr bool Reduce = ReduceT::value, hasSd = !std::is_empty_v<SrcData>;
    constexpr int CAP = leafCap, GL = G;
    using V4 = sycl::vec<Real, 4>;
    const Kernel k = kernel_;
    const interp::Orders o = orders_;
    const int nodes = o.nodes();
    const std::int32_t *box = leafBox_.data(), *slotOf = leafSlot_.data(), *el = coneEl_.data(), *lev = sLevel_.data();
    const std::uint32_t *sb = sBegin_.data(), *se = sEnd_.data();
    const V4 *rec = srcRec_.data(), *nrm = srcNrm_.data();
    const Real* rwi = srcWi_.data();
    const cones::FaceGrid<Real>* grids = grids_.data();
    const Real *H = sH_.data(), *cheb = cheb_.data();
    const int c0 = chebOff_[0], c1 = chebOff_[1], c2 = chebOff_[2];
    const interp::DeviceTransformTables<Real> tab = dtab_;
    Real* coef = coef_.data();
    q_.submit([&](sycl::handler& h) {
      sycl::local_accessor<Real, 1> buf{sycl::range<1>{static_cast<std::size_t>(nodes) * W}, h};
      sycl::local_accessor<V4, 1> lp{sycl::range<1>{CAP}, h};
      sycl::local_accessor<Real, 1> lwi{sycl::range<1>{CAP}, h};
      sycl::local_accessor<Real, 1> lss{sycl::range<1>{CAP}, h};
      sycl::local_accessor<V4, 1> ln{sycl::range<1>{hasSd ? CAP : 1}, h};
      sycl::local_accessor<Real, 1> lsn{sycl::range<1>{hasSd ? CAP : 1}, h};
      h.parallel_for(sycl::nd_range<1>{numLeafWork_ * GL, GL}, [=](sycl::nd_item<1> it) {
        const auto grp = it.get_group();
        const int tid = static_cast<int>(it.get_local_id(0));
        const std::int32_t b = box[it.get_group(0)], slot = slotOf[it.get_group(0)];
        const cones::FaceGrid<Real> g = grids[lev[b]];
        const Real Hb = H[lev[b]];
        int face, iw, iu, iv;
        g.decode(el[slot], face, iw, iu, iv);
        const std::uint32_t s0 = sb[b], ns = se[b] - s0;
        Real* data = &buf[0];
        for (int i = tid; i < nodes * W; i += GL) data[i] = Real(0);
        constexpr Real fourPi = Real(12.5663706143591729539);
        for (std::uint32_t base = 0; base < ns; base += CAP) {
          const int cnt = static_cast<int>(ns - base < CAP ? ns - base : CAP);
          sycl::group_barrier(grp);
          for (int j = tid; j < cnt; j += GL) {
            const V4 r = rec[s0 + base + j];
            lp[j] = r;
            lwi[j] = rwi[s0 + base + j];
            lss[j] = r.x() * r.x() + r.y() * r.y() + r.z() * r.z();
            if constexpr (hasSd) {
              const V4 n = nrm[s0 + base + j];
              ln[j] = n;
              lsn[j] = r.x() * n.x() + r.y() * n.y() + r.z() * n.z();
            }
          }
          sycl::group_barrier(grp);
          {
            for (int dir = tid; dir < NB; dir += GL) {
              const int j = dir % NU, kq = dir / NU;
              const Real up = g.uOf(iu, cheb[c1 + j]), vp = g.uOf(iv, cheb[c2 + kq]);
              const Vec3f t1 = cones::fromFace(face, Real(1), up, vp, Real(1));  // n + u e_b + v e_c
              const Real len = math::sqrt(norm2(t1));
              const Vec3f uh = t1 * (Real(1) / len);
              Real r[NW], m2r[NW], r2[NW], i2r[NW], ar[NW], ai[NW];
#pragma unroll
              for (int i = 0; i < NW; ++i) {
                r[i] = Hb / g.wOf(iw, iu, iv, cheb[c0 + i]) * len;
                m2r[i] = Real(-2) * r[i], r2[i] = r[i] * r[i], i2r[i] = Real(0.5) / r[i], ar[i] = ai[i] = Real(0);
              }
#pragma unroll 2
              for (int m = 0; m < cnt; ++m) {
                const V4 P = lp[m];
                const Real wi = lwi[m], ss = lss[m];
                const Real pr = P.x() * uh.x + P.y() * uh.y + P.z() * uh.z;
                Real sn = Real(0), un = Real(0);
                if constexpr (hasSd) {
                  const V4 N = ln[m];
                  sn = lsn[m], un = N.x() * uh.x + N.y() * uh.y + N.z() * uh.z;
                }
#pragma unroll
                for (int i = 0; i < NW; ++i) {
                  const Real A = m2r[i] * pr + ss;
                  const Real d2 = r2[i] + A, dinv = math::fast::rsqrt(d2);
                  const Real dpr = d2 * dinv + r[i], d0 = dpr + m2r[i];
                  const Real delta = (A - d0 * dpr) * i2r[i] + d0;
                  Value e = Reduce ? k.expikFastReduced(delta) : k.expikFast(delta);
                  if constexpr (hasSd) e = kernels::detail::cmul(e, k.rayFactor(dinv, (sn - r[i] * un) * dinv));
                  const Real er = e.real() * dinv, ei = e.imag() * dinv;
                  ar[i] += P.w() * er;
                  ar[i] -= wi * ei;
                  ai[i] += P.w() * ei;
                  ai[i] += wi * er;
                }
              }
#pragma unroll
              for (int i = 0; i < NW; ++i) {
                Real* d = data + (i + NW * dir) * W;  // dc / d = r * dinv: r applied here
                d[0] += fourPi * r[i] * ar[i];
                d[1] += fourPi * r[i] * ai[i];
              }
            }
          }
        }
        sycl::group_barrier(grp);
        interp::transformInGroup<Real, Real, NW, NU>(it, data, o, tab, W);
        Real* dst = coef + static_cast<std::size_t>(slot) * nodes * W;
        for (int kk = tid; kk < nodes * W; kk += GL) dst[kk] = data[kk];
      });
    });
  }

  // Upward kernel for parent level l (see the file header)
  void upMain(int l)
  {
    using namespace face_detail;
    const std::int32_t g0 = upGroupLevelBegin_[l], g1 = upGroupLevelBegin_[l + 1];
    if (g1 == g0) return;
    const std::size_t nG = static_cast<std::size_t>(g1 - g0);
    const cones::FaceGrid<Real> gp = cones_.grid[l];
    const Real Hp = static_cast<Real>(S_.H[l]), Hc = static_cast<Real>(S_.H[l + 1]);
    const Kernel kf = kernel_;
    const ULo ulc = gcLo_(l);
    std::array<Real, NW> c6;
    std::array<Real, 8> c8{};  // padded to 8 (thread maps assume up to 8 nodes per angular axis)
    for (int i = 0; i < NW; ++i) c6[i] = static_cast<Real>(interp::chebNode(i, NW));
    for (int i = 0; i < NU; ++i) c8[i] = static_cast<Real>(interp::chebNode(i, NU));
    const Real* cf = coef_.data();
    Real* coefOut = coef_.data();
    const UpGroup* gs = upGroups_.data() + g0;
    const std::int32_t* els = upElem_.data() + static_cast<std::size_t>(g0) * UPCAP;
    const std::int32_t* gsl = upGSlot_.data() + static_cast<std::size_t>(g0) * UPCAP;
    const UpStaged* sts = upStaged_.data();
    const std::uint32_t* tks = upTasks_.data();
    const cones::FaceLine* lns = upLines_.data() + lineLevelBegin_[l];
    const cones::FaceLineNode* lnn = upLineNodes_.data();
    const Real* chebU = cheb_.data() + chebOff_[1];
    const interp::Orders oc = orders_;
    const interp::DeviceTransformTables<Real> tabc = dtab_;
    constexpr int CAP = UPCAP, RB = (NB + 15) / 16, PF = (CW + UG - 1) / UG;
    constexpr int SAS = NB * 2 + 2;  // per half-warp radial scratch stride: +2 words, so the two half-warps of a warp
                                     // use different banks (stride NB * 2 = 128 gave 2-way conflicts on every access)
    using V2 = sycl::vec<Real, 2>;
    q_.submit([&](sycl::handler& h) {
      sycl::local_accessor<Real, 1> sC{sycl::range<1>{2 * CW}, h};
      sycl::local_accessor<Real, 1> sA{sycl::range<1>{16 * SAS}, h};
      sycl::local_accessor<Real, 1> sAcc{sycl::range<1>{CAP * NODES * 2}, h};
      sycl::local_accessor<Real, 1> sW{sycl::range<1>{CAP * NW}, h};
      sycl::local_accessor<std::int32_t, 1> sEl{sycl::range<1>{CAP}, h};
      h.parallel_for(sycl::nd_range<1>{nG * UG, UG}, [=](sycl::nd_item<1> ndi) {
        // AdaptiveCpp keeps the closure on the stack: copy what the loops use into locals (registers)
        const cones::FaceGrid<Real> gpl = gp;
        const ULo ul = ulc;
        const Kernel kl = kf;
        const Real Hpl = Hp, Hcl = Hc;
        const Real* const cfl = cf;
        Real* const outl = coefOut;
        const UpStaged* const stl = sts;
        const std::uint32_t* const tkl = tks;
        const cones::FaceLine* const lnl = lns;
        const cones::FaceLineNode* const lnnl = lnn;
        const Real* const chU = chebU;
        const int tid = static_cast<int>(ndi.get_local_id(0));
        const auto grp = ndi.get_group();
        const auto sg = ndi.get_sub_group();
        const bool warpSync = sg.get_max_local_range()[0] >= 32;
        const std::size_t gi = ndi.get_group(0);
        const UpGroup g = gs[gi];
        const int hw = tid >> 4, l16 = tid & 15, j = (tid >> 1) & 7, kh = tid & 1;
        const Real cj = c8[j];
        Real ck[4];
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) ck[kk] = c8[kh * 4 + kk];
        for (int x = tid; x < CAP * NODES * 2; x += UG) sAcc[x] = Real(0);
        if (tid < CAP) sEl[tid] = els[gi * CAP + tid];
        if (tid < CAP * NW) {
          const std::int32_t e = els[gi * CAP + tid / NW];
          sW[tid] = e >= 0 ? gpl.wOf(e & 1023, (e >> 10) & 1023, (e >> 20) & 1023, c6[tid % NW]) : Real(1);
        }
        auto tpos = [](int x) {
          const int a = x >> 1, a0 = a % NW, bg = a / NW;
          return ((a0 * NB + bg) << 1) | (x & 1);
        };
        if (g.stBegin < g.stEnd)
          for (int x = tid; x < CW; x += UG) sC[tpos(x)] = cfl[static_cast<std::size_t>(stl[g.stBegin].slot) * CW + x];
        sycl::group_barrier(grp);
        int buf = 0;
        Real* sAh = &sA[hw * SAS];
        auto ld2 = [](const Real* p) { return *reinterpret_cast<const V2*>(p); };
        for (int s = g.stBegin; s < g.stEnd; ++s) {
          const UpStaged st = stl[s];
          Real pf[PF];
          const bool hasNext = s + 1 < g.stEnd;
          if (hasNext) {
            const Real* src = cfl + static_cast<std::size_t>(stl[s + 1].slot) * CW;
#pragma unroll
            for (int r = 0; r < PF; ++r) pf[r] = (CW % UG == 0 || tid + UG * r < CW) ? src[tid + UG * r] : Real(0);
          }
          const Real* C = &sC[buf * CW];
          const Real hc = Hcl * Real(0.5);
          const Vec3f ov{(st.oct & 1) ? hc : -hc, (st.oct & 2) ? hc : -hc, (st.oct & 4) ? hc : -hc};
          const int nS = st.tEnd - st.tBegin, nt = nS + (st.lEnd - st.lBegin);
          for (int it = 0; it < nt; it += 16) {
            const int t = it + hw;
            const bool on = t < nt, isLine = on && t >= nS;
            std::uint32_t task = 0;
            int e = 0, i = 0;
            Real w = Real(1);
            cones::LayerMap<Real> m{Real(1), Real(0), Real(0), Real(0)};
            cones::FaceLine ln{0, 0, 0, Real(0), Real(0)};
            if (on) {
              Real xw;
              if (!isLine) {
                task = tkl[st.tBegin + t];
                e = static_cast<int>(task >> 20), i = static_cast<int>((task >> 16) & 15);
                w = sW[e * NW + i];
                m = cones::layerMap<Real>(g.face, w, Hpl, Hcl, st.oct);
                xw = (m.wc - st.wC) * st.wHinv;
              } else {
                task = tkl[st.lBegin + (t - nS)];
                e = static_cast<int>(task >> lineShift);
                ln = lnl[task & ((1u << lineShift) - 1)];
                xw = ln.xiW;
              }
              Real T[NW];
              chebT(xw, NW, T);
#pragma unroll
              for (int r = 0; r < RB; ++r) {
                const int bg = l16 + 16 * r;
                if (NB % 16 != 0 && bg >= NB) break;
                Real re = Real(0), im = Real(0);
#pragma unroll
                for (int a0 = 0; a0 < NW; ++a0) {
                  const V2 c = ld2(C + (a0 * NB + bg) * 2);
                  re += T[a0] * c.x();
                  im += T[a0] * c.y();
                }
                *reinterpret_cast<V2*>(sAh + bg * 2) = V2{re, im};
              }
            }
            if (warpSync)
              sycl::group_barrier(sg);
            else
              sycl::group_barrier(grp);
            if (!isLine) {
              const int j0 = static_cast<int>((task >> 12) & 15), j1 = static_cast<int>((task >> 8) & 15);
              const int k0 = static_cast<int>((task >> 4) & 15), k1 = static_cast<int>(task & 15);
              if (on && j >= j0 && j < j1) {
                const std::int32_t el = sEl[e];
                const int iu = (el >> 10) & 1023, iv = (el >> 20) & 1023;
                const Real invDen = Real(1) / m.den;
                const Real up = gpl.uOf(iu, cj);
                const Real uc = (up - m.s1) * invDen;
                Real Tb[NU];
                chebT(gcUlocal(uc, st.iuc, ul), NU, Tb);
                Real B[NU][2];
#pragma unroll
                for (int gm = 0; gm < NU; ++gm) {
                  Real re = Real(0), im = Real(0);
#pragma unroll
                  for (int b = 0; b < NU; ++b) {
                    const V2 c = ld2(sAh + (b + NU * gm) * 2);
                    re += Tb[b] * c.x();
                    im += Tb[b] * c.y();
                  }
                  B[gm][0] = re, B[gm][1] = im;
                }
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                  const int k = kh * 4 + kk;
                  if (k < k0 || k >= k1) continue;
                  const Real vp = gpl.uOf(iv, ck[kk]);
                  const Real vc = (vp - m.s2) * invDen;
                  Real Tg[NU];
                  chebT(gcUlocal(vc, st.ivc, ul), NU, Tg);
                  Real re = Real(0), im = Real(0);
#pragma unroll
                  for (int gm = 0; gm < NU; ++gm) {
                    re += Tg[gm] * B[gm][0];
                    im += Tg[gm] * B[gm][1];
                  }
                  const auto tr = kl.transferFast(cones::fromFace(g.face, w, up, vp, Hpl), ov);
                  Real* a = &sAcc[(e * NODES + i + NW * (j + NU * k)) * 2];
                  a[0] += re * tr.real() - im * tr.imag();
                  a[1] += re * tr.imag() + im * tr.real();
                }
              }
            } else if (l16 < (ln.axisN & 255)) {
              // fallback line: contract the shared angular axis at xiA, then sum over the varying one at xi
              const cones::FaceLineNode nd = lnnl[ln.nBegin + l16];
              const bool sharedU = (ln.axisN >> 8) == 1;
              Real Ta[NU], Tv[NU];
              chebT(ln.xiA, NU, Ta);
              chebT(nd.xi, NU, Tv);
              Real re = Real(0), im = Real(0);
#pragma unroll
              for (int q = 0; q < NU; ++q) {
                Real br = Real(0), bi = Real(0);
#pragma unroll
                for (int r = 0; r < NU; ++r) {
                  const int bg = sharedU ? r + NU * q : q + NU * r;  // shared index r, varying index q
                  const V2 c = ld2(sAh + bg * 2);
                  br += Ta[r] * c.x();
                  bi += Ta[r] * c.y();
                }
                re += Tv[q] * br;
                im += Tv[q] * bi;
              }
              const int ni = nd.node % NW, nj = (nd.node / NW) % NU, nk = nd.node / (NW * NU);
              const std::int32_t el = sEl[e];
              const Real wn = sW[e * NW + ni];
              const Vec3f xp = cones::fromFace(g.face, wn, gpl.uOf((el >> 10) & 1023, chU[nj]), gpl.uOf((el >> 20) & 1023, chU[nk]), Hpl);
              const auto tr = kl.transferFast(xp, ov);
              Real* a = &sAcc[(e * NODES + nd.node) * 2];
              a[0] += re * tr.real() - im * tr.imag();
              a[1] += re * tr.imag() + im * tr.real();
            }
            if (warpSync)
              sycl::group_barrier(sg);
            else
              sycl::group_barrier(grp);
          }
          if (hasNext) {
            Real* dst = &sC[(buf ^ 1) * CW];
#pragma unroll
            for (int r = 0; r < PF; ++r)
              if (CW % UG == 0 || tid + UG * r < CW) dst[tpos(tid + UG * r)] = pf[r];
          }
          sycl::group_barrier(grp);
          buf ^= 1;
        }
        // fused transform: values -> coefficients for the group's g.n elements, three in-place passes over all
        // elements at once (one line per thread item), then one coalesced write per element
        {
          const interp::Orders ol = oc;
          const interp::DeviceTransformTables<Real> tb = tabc;
          Real* acc = &sAcc[0];
          constexpr int I0 = NB * 2, I12 = NODES / NU * 2;  // line items per element and pass
          for (int x = tid; x < g.n * I0; x += UG) {
            const int e = x / I0;
            interp::transformPassN<Real, Real, NW>(acc + e * CW, ol, 0, tb.M[0], 2, x - e * I0, I0);
          }
          sycl::group_barrier(grp);
          for (int x = tid; x < g.n * I12; x += UG) {
            const int e = x / I12;
            interp::transformPassN<Real, Real, NU>(acc + e * CW, ol, 1, tb.M[1], 2, x - e * I12, I12);
          }
          sycl::group_barrier(grp);
          for (int x = tid; x < g.n * I12; x += UG) {
            const int e = x / I12;
            interp::transformPassN<Real, Real, NU>(acc + e * CW, ol, 2, tb.M[2], 2, x - e * I12, I12);
          }
          sycl::group_barrier(grp);
        }
        for (int x = tid; x < g.n * CW; x += UG) {
          const int e = x / CW;
          outl[static_cast<std::size_t>(gsl[gi * CAP + e]) * CW + (x - e * CW)] = sAcc[x];
        }
      });
    });
  }

  // child grid u-local coordinate without the grid object in the kernel: (u - lo(iu)) / (hu / 2)
  struct ULo {
    Real hu, inv;  // cell width, 2 / cell width
    int ext;
  };
  ULo gcLo_(int l) const { return {cones_.grid[l + 1].hu, Real(2) / cones_.grid[l + 1].hu, cones_.grid[l + 1].ext}; }
  static IFGF_INLINE Real gcUlocal(Real u, int iu, ULo g)
  {
    return (u - (Real(-1) + (Real(iu - g.ext) + Real(0.5)) * g.hu)) * g.inv;
  }

  // Far field for the target boxes of level l: one work-group per chunk of up to G * FarR targets of one box
  void far(int l)
  {
    using namespace face_detail;
    const std::int32_t first = farLevelBegin_[l];
    const std::size_t count = static_cast<std::size_t>(farLevelBegin_[l + 1] - first);
    constexpr std::size_t QC = static_cast<std::size_t>(farChunkSize);  // queue capacity = targets per chunk
    q_.submit([&](sycl::handler& h) {
      sycl::local_accessor<sycl::vec<Real, 4>, 1> lv{sycl::range<1>{static_cast<std::size_t>(farNSC * farCS / 4)}, h};
      sycl::local_accessor<std::int32_t, 1> lel{sycl::range<1>{static_cast<std::size_t>(farConeListCapacity)}, h};
      sycl::local_accessor<Real, 1> lq{sycl::range<1>{(QC + farNSC) * 5}, h};  // per queue entry: xi[3], cf (re, im)
      sycl::local_accessor<std::int32_t, 1> lqt{sycl::range<1>{QC + farNSC + 2 * farNSC}, h};  // target per entry; counters
      sycl::local_accessor<Real, 1> lacc{sycl::range<1>{QC * 2}, h};        // per target of the chunk
      sycl::local_accessor<std::int32_t, 1> lred{sycl::range<1>{2 * static_cast<std::size_t>(G)}, h};  // group min
      const Kernel k = kernel_;
      const interp::Orders o = orders_;
      const std::int32_t *box = farBox_.data() + first, *chunkBegin = farBegin_.data() + first;
      const std::int32_t *off = farOff_.data(), *el = coneEl_.data();
      const FarPairF<Real>* pairs = farPairs_.data();
      const std::uint32_t* te = tEnd_.data();
      const Vec3f* tp = tgtPts_.data();
      const Real* coef = coef_.data();
      Value* res = result_.data();
      int* miss = misses_.data();
      const Vec3f* tpl = tgtPtsLo_.data();
      const Vec3f* cpl = farCenterLo_.data();
      h.parallel_for(sycl::nd_range<1>{count * G, G}, [=](sycl::nd_item<1> it) {
        const std::int32_t t = box[it.get_group(0)];
        const int b0 = chunkBegin[it.get_group(0)];
        const int m = sycl::min(static_cast<int>(te[t]) - b0, G * FarR);
        FarLocal L{reinterpret_cast<Real*>(&lv[0]), &lel[0], &lq[0], &lqt[0], &lacc[0], &lred[0]};
        if (m > G)
          farChunk<2>(it, L, k, o, t, b0, b0 + m, off, el, pairs, tp, coef, res, miss, tpl, cpl);
        else
          farChunk<1>(it, L, k, o, t, b0, b0 + m, off, el, pairs, tp, coef, res, miss, tpl, cpl);
      });
    });
  }

  struct FarLocal {
    Real* lc;              // staged cone coefficients
    std::int32_t* lcones;  // the source box's cone keys
    Real* q;               // queue: xi[3], cf[2] per entry
    std::int32_t* qt;      // queue: target index per entry; qt[farChunkSize] = count
    Real* acc;             // per target accumulator (re, im)
    std::int32_t* red;     // group min scratch, 2 x (number of sub-groups), double buffered
  };

  // One chunk of <= G * R targets of one target box against all its far pairs. Per far pair every target finds its
  // cone (natural face element) in the source box; per distinct cone (group minimum): stage the cone, append the
  // targets that need it to a local queue, evaluate the queue two entries per thread (same cone: shared coefficient
  // loads), accumulate per target in local memory. Warps without queue entries skip the evaluation.
  // The centred factor is computed in compensated FP32 from hi/lo positions (tp + tpl, centre + cpl), so its phase
  // error is a few eps in absolute terms instead of ~eps kappa |y|.
  template <int R>
  static IFGF_INLINE void farChunk(const sycl::nd_item<1>& it, const FarLocal& L, const Kernel& k, const interp::Orders& o,
                                   std::int32_t t, int begin, int end, const std::int32_t* off, const std::int32_t* el,
                                   const face_detail::FarPairF<Real>* pairs, const Vec3f* tp, const Real* coef, Value* res,
                                   int* miss, const Vec3f* tpl, const Vec3f* cpl)
  {
    constexpr std::int32_t done = 0x7fffffff;
    constexpr int QC = farChunkSize;
    const int tid = static_cast<int>(it.get_local_id(0));
    const auto grp = it.get_group();
    Real* const lc = L.lc;
    std::int32_t* const lcones = L.lcones;
    Real* const q = L.q;
    std::int32_t* const qt = L.qt;
    Real* const lacc = L.acc;
    Vec3f x[R];
#pragma unroll
    for (int j = 0; j < R; ++j) {
      const int p = begin + tid + j * G;
      x[j] = tp[p < end ? p : begin];
    }
    for (int i = tid; i < 2 * QC; i += G) lacc[i] = Real(0);
    // group minimum: sub-group reduction, then the sub-group results through double-buffered local scratch (one
    // barrier per call; AdaptiveCpp's reduce_over_group(group) races when called back to back without a barrier)
    const auto sg = it.get_sub_group();
    const int sgs = static_cast<int>(sg.get_local_range()[0]), nsg = G / sgs, sgi = tid / sgs;
    int redCall = 0;
    auto groupMin = [&](std::int32_t v) {
      v = sycl::reduce_over_group(sg, v, sycl::minimum<std::int32_t>());
      std::int32_t* r = L.red + (redCall & 1) * nsg;
      ++redCall;
      if (static_cast<int>(sg.get_local_id()[0]) == 0) r[sgi] = v;
      sycl::group_barrier(grp);
      std::int32_t m = r[0];
      for (int i = 1; i < nsg; ++i) m = sycl::min(m, r[i]);
      return m;
    };
    for (auto n = off[t]; n < off[t + 1]; ++n) {
      const face_detail::FarPairF<Real> pr = pairs[n];
      const std::int32_t numCones = pr.coneEnd - pr.coneBegin;
      const bool staged = numCones <= farConeListCapacity;
      if (staged) {
        sycl::group_barrier(grp);
        for (int i = tid; i < numCones; i += G) lcones[i] = el[pr.coneBegin + i];
        sycl::group_barrier(grp);
      }
      std::int32_t slot[R];
      Real xi[R][3];
      Value cf[R];
#pragma unroll
      for (int j = 0; j < R; ++j) {
        slot[j] = done;
        xi[j][0] = xi[j][1] = xi[j][2] = Real(0);
        cf[j] = Value(0);
        if (begin + tid + j * G < end) {
          const Vec3f y = x[j] - pr.center;
          const cones::FaceCoord<Real> fc = cones::faceOf(y, pr.H);
          const std::int32_t e = cones::locateFace(pr.grid, fc, xi[j]);
          std::int32_t sl;
          if (staged) {
            sl = face_detail::findSlot(lcones, 0, numCones, e);
            if (sl >= 0) sl += pr.coneBegin;
          } else {
            sl = face_detail::findSlot(el, pr.coneBegin, pr.coneEnd, e);
          }
          if (sl < 0) {
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device>(*miss).fetch_add(1);
          } else {
            slot[j] = sl;
            Vec3f yh, yl;
            const Vec3f xl = tpl[begin + tid + j * G], cl = cpl[n];
            comp::twoSum(x[j].x, -pr.center.x, yh.x, yl.x);
            comp::twoSum(x[j].y, -pr.center.y, yh.y, yl.y);
            comp::twoSum(x[j].z, -pr.center.z, yh.z, yl.z);
            yl = yl + (xl - cl);
            cf[j] = k.centeredComp(yh, yl);
          }
        }
      }
      for (;;) {  // rounds: up to farNSC distinct cones at once
        std::int32_t Ld[farNSC];
        int nL = 0;
        std::int32_t prev = -1;
#pragma unroll
        for (int r = 0; r < farNSC; ++r) {
          std::int32_t mine = done;
#pragma unroll
          for (int j = 0; j < R; ++j) mine = sycl::min(mine, slot[j] > prev ? slot[j] : done);
          const std::int32_t ld = groupMin(mine);
          Ld[r] = ld;
          if (ld == done) break;
          prev = ld, ++nL;
        }
        if (nL == 0) break;
        std::int32_t* const cnt = qt + QC + farNSC;  // [farNSC] counts, then [farNSC] fill
        sycl::group_barrier(grp);                    // previous round: staged cones and the queue are free
        for (int r = 0; r < nL; ++r) {
          const Real* srcCoef = coef + static_cast<std::size_t>(Ld[r]) * CW;
          Real* dstC = lc + r * farCS;
          if constexpr (CW % 4 == 0) {
            using V4 = sycl::vec<Real, 4>;
            constexpr int C4 = CW / 4, NL = (C4 + G - 1) / G;
            const V4* s4 = reinterpret_cast<const V4*>(srcCoef);
            V4* l4 = reinterpret_cast<V4*>(dstC);
            V4 tmp[NL];
#pragma unroll
            for (int q2 = 0; q2 < NL; ++q2)
              if (C4 % G == 0 || tid + G * q2 < C4) tmp[q2] = s4[tid + G * q2];
#pragma unroll
            for (int q2 = 0; q2 < NL; ++q2)
              if (C4 % G == 0 || tid + G * q2 < C4) l4[tid + G * q2] = tmp[q2];
          } else {
            for (int i = tid; i < CW; i += G) dstC[i] = srcCoef[i];
          }
        }
        if (tid < 2 * farNSC) cnt[tid] = 0;
        sycl::group_barrier(grp);
        int rk[R];
#pragma unroll
        for (int j = 0; j < R; ++j) {
          rk[j] = -1;
#pragma unroll
          for (int r = 0; r < farNSC; ++r)
            if (r < nL && slot[j] == Ld[r]) rk[j] = r;
          if (rk[j] >= 0)
            sycl::atomic_ref<std::int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                             sycl::access::address_space::local_space>(cnt[rk[j]])
                .fetch_add(1);
        }
        sycl::group_barrier(grp);
        int offs[farNSC + 1];  // bucket starts, padded to even lengths
        offs[0] = 0;
#pragma unroll
        for (int r = 0; r < farNSC; ++r) offs[r + 1] = offs[r] + (r < nL ? (cnt[r] + 1) & ~1 : 0);
#pragma unroll
        for (int j = 0; j < R; ++j)
          if (rk[j] >= 0) {
            int sel = 0;
#pragma unroll
            for (int r = 0; r < farNSC; ++r)
              if (r == rk[j]) sel = offs[r];
            const int e = sel + sycl::atomic_ref<std::int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                                 sycl::access::address_space::local_space>(cnt[farNSC + rk[j]])
                                    .fetch_add(1);
            q[e * 5 + 0] = xi[j][0], q[e * 5 + 1] = xi[j][1], q[e * 5 + 2] = xi[j][2];
            q[e * 5 + 3] = cf[j].real(), q[e * 5 + 4] = cf[j].imag();
            qt[e] = tid + j * G;
            slot[j] = done;
          }
        if (tid < nL && (cnt[tid] & 1)) {  // pad entry of an odd bucket
          int sel = 0;
#pragma unroll
          for (int r = 0; r < farNSC; ++r)
            if (r == tid) sel = offs[r];
          const int e = sel + cnt[tid];
          qt[e] = -1;
          q[e * 5 + 0] = q[e * 5 + 1] = q[e * 5 + 2] = Real(0);
        }
        sycl::group_barrier(grp);
        const int total = offs[farNSC];
        for (int pp = tid; 2 * pp < total; pp += G) {
          const int e0 = 2 * pp, e1 = 2 * pp + 1;
          int r = 0;
#pragma unroll
          for (int rr = 1; rr < farNSC; ++rr)
            if (e0 >= offs[rr]) r = rr;
          Real xs[2][3], v[2][W];
#pragma unroll
          for (int d = 0; d < 3; ++d) xs[0][d] = q[e0 * 5 + d], xs[1][d] = q[e1 * 5 + d];
          interp::detail::evalAtPoints<W, 2, NW>(lc + r * farCS, xs, v, o);
#pragma unroll
          for (int jj = 0; jj < 2; ++jj) {
            const int e = jj == 0 ? e0 : e1;
            const int tg = qt[e];
            if (tg < 0) continue;
            const Value rv = kernels::detail::cmul(Value(v[jj][0], v[jj][1]), Value(q[e * 5 + 3], q[e * 5 + 4]));
            Real* a = lacc + 2 * tg;
            a[0] += rv.real();
            a[1] += rv.imag();
          }
        }
      }
    }
    sycl::group_barrier(grp);
#pragma unroll
    for (int j = 0; j < R; ++j) {
      const int p = begin + tid + j * G;
      if (p < end) res[p] += Value(lacc[2 * (tid + j * G)], lacc[2 * (tid + j * G) + 1]);
    }
  }

  sycl::queue& q_;
  FacePlanOptions opt_;
  interp::Orders orders_;
  Tree S_, T_;
  InteractionLists lists_;
  cones::FaceConeTables<Real> cones_;
  Kernel kernel_;
  FacePlanStats stats_;

  exec::DeviceBuffer<Vec3f> srcPts_, tgtPts_, tgtLoc_;
  exec::DeviceBuffer<face_detail::NearPairC> nearPairs_;
  exec::DeviceBuffer<Vec3f> srcPtsLo_, tgtLocLo_;  // lo parts of the leaf-local positions (near field)
  exec::DeviceBuffer<sycl::vec<Real, 4>> srcNrmLo_;  // lo parts of the normals
  exec::DeviceBuffer<sycl::vec<Real, 4>> srcRec_, srcNrm_;
  exec::DeviceBuffer<Real> srcWi_;
  exec::DeviceBuffer<std::int32_t> sLevel_, nearOff_, farOff_, coneOff_, coneEl_;
  exec::DeviceBuffer<std::uint32_t> sBegin_, sEnd_, tBegin_, tEnd_;
  exec::DeviceBuffer<Real> sH_, cheb_, tab_, coef_;
  exec::DeviceBuffer<cones::FaceGrid<Real>> grids_;
  exec::DeviceBuffer<face_detail::FarPairF<Real>> farPairs_;
  exec::DeviceBuffer<std::int32_t> leafBox_, leafSlot_, farBox_, farBegin_, nearBox_;
  std::vector<cones::FaceLevelMapDevice> devMaps_;  // level maps per parent level
  exec::DeviceBuffer<face_detail::UpGroup> upGroups_;
  exec::DeviceBuffer<std::int32_t> upElem_, upGSlot_;
  exec::DeviceBuffer<face_detail::UpStaged> upStaged_;
  exec::DeviceBuffer<std::uint32_t> upTasks_;
  exec::DeviceBuffer<cones::FaceLine> upLines_;
  exec::DeviceBuffer<cones::FaceLineNode> upLineNodes_;
  exec::DeviceBuffer<Value> w_, result_;
  exec::DeviceBuffer<int> misses_;
  interp::DeviceTransformTables<Real> dtab_{};
  int chebOff_[3] = {0, 0, 0};
  std::vector<std::int32_t> upSlotLevelBegin_, upGroupLevelBegin_, lineLevelBegin_, farLevelBegin_;
  std::size_t numLeafWork_ = 0, numNearWork_ = 0;
  bool leafPhaseSmall_ = false;
  exec::DeviceBuffer<Vec3f> tgtPtsLo_, farCenterLo_;  // lo parts of the targets and source centres (relative to c0)
};

}  // namespace ifgf
