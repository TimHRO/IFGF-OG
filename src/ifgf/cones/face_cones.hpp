#pragma once

// Face-chart cones (projective cube-face cones). Used by FacePlan (plan/face_plan.hpp); built on the device by
// buildFaceConesDevice (cones/face_activation_device.hpp).
//
// Coordinates of y = point - box center (box side H) in the chart of face f = 2a + (sign < 0), normal
// n = sign * e_a, in-plane axes e_b, e_c (b = (a+1)%3, c = (a+2)%3):
//   w = H / (y.n),  u = (y.e_b) / (y.n),  v = (y.e_c) / (y.n)
// A point's natural face is the one of its largest |component| (|u|, |v| <= 1). For a child center offset d
// (child side Hc), in the SAME chart:
//   den = 1 - (d.n) w / H,  w_c = (Hc/H) w / den,  u_c = (u - d_b w / H) / den,  v_c = (v - d_c w / H) / den
// i.e. per radial layer (fixed w) an axis-aligned affine map in u and v separately (exact).
//
// Grids per level: n_w = max(nwMin, round(fw m)), n_u = n_v = max(nuMin, round(fu m)) elements per face,
// m = ceil(max(density H, 1)); u, v cells of width 2/n_u over [-1, 1], plus `ext` cells per side (charts are
// extended for parent nodes that a parent maps beyond the face edge); w in [wlo, 1] for every column (default,
// FaceParams::col = false: nested cube shells ||y||_inf = H / w) or in [wlo, smax sqrt(1 + U^2 + V^2)] per (u, v)
// column (col = true; U, V: the largest |u|, |v| of the column), n_w uniform cells (close to uniform in s).
// Element key (int32): ((face * nw + iw) * span + iu) * span + iv, span = n_u + 2 ext.
//
// Consistency: the far field computes faces and elements with these Real functions; activation computes the same
// and activates neighbours within a margin of every border (also the other face when two components nearly tie).
// The upward pass needs no margins: the level maps decide which child element each parent node uses (tasks with
// node ranges, fallback entries with xi), the upward kernel only evaluates.

#include <ifgf/core/math.hpp>
#include <ifgf/core/vec3.hpp>
#include <ifgf/interp/orders.hpp>
#include <ifgf/interp/tables.hpp>
#include <ifgf/lists/lists.hpp>
#include <ifgf/tree/tree.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace ifgf::cones {

template <class Real>
struct FaceGrid {
  int nw = 1, nu = 1, ext = 0, span = 1, col = 1;
  Real wlo = 0, hu = 2, smax = 1, hw0 = 1;

  Real uLo(int iu) const
  {
    IFGF_NO_FP_CONTRACT
    return Real(-1) + Real(iu - ext) * hu;
  }
  Real uAbsMax(int iu) const
  {
    IFGF_NO_FP_CONTRACT
    const Real a = uLo(iu), b = a + hu;
    return math::fmax(math::fmax(a, -a), math::fmax(b, -b));
  }
  Real hw(int iu, int iv) const
  {
    IFGF_NO_FP_CONTRACT
    if (!col) return hw0;
    const Real U = uAbsMax(iu), V = uAbsMax(iv);
    return (smax * math::sqrt(Real(1) + U * U + V * V) - wlo) / Real(nw);
  }
  Real wOf(int iw, int iu, int iv, Real xi) const
  {
    IFGF_NO_FP_CONTRACT
    return wlo + (Real(iw) + Real(0.5) + Real(0.5) * xi) * hw(iu, iv);
  }
  Real uOf(int iu, Real xi) const
  {
    IFGF_NO_FP_CONTRACT
    return Real(-1) + (Real(iu - ext) + Real(0.5) + Real(0.5) * xi) * hu;
  }
  Real wLocal(Real w, int iw, int iu, int iv) const
  {
    IFGF_NO_FP_CONTRACT
    const Real h = hw(iu, iv);
    return (w - (wlo + (Real(iw) + Real(0.5)) * h)) / (Real(0.5) * h);
  }
  // with the column's hw given (callers with a table of hw per column)
  Real wLocalH(Real w, int iw, Real h) const
  {
    IFGF_NO_FP_CONTRACT
    return (w - (wlo + (Real(iw) + Real(0.5)) * h)) / (Real(0.5) * h);
  }
  Real uLocal(Real u, int iu) const
  {
    IFGF_NO_FP_CONTRACT
    return (u - (Real(-1) + (Real(iu - ext) + Real(0.5)) * hu)) / (Real(0.5) * hu);
  }
  // extended-chart index of u (may be outside [0, span))
  int uIndex(Real u) const
  {
    IFGF_NO_FP_CONTRACT
    return static_cast<int>(math::floor((u + Real(1)) / hu)) + ext;
  }
  // index of u in the primary chart (clamped to [ext, ext + nu)): natural-face points
  int uIndexPrimary(Real u) const
  {
    IFGF_NO_FP_CONTRACT
    const Real q = math::fmin(math::fmax(math::floor((u + Real(1)) / hu), Real(0)), Real(nu - 1));
    return static_cast<int>(q) + ext;
  }
  int wIndex(Real w, int iu, int iv) const
  {
    IFGF_NO_FP_CONTRACT
    const Real q = math::fmin(math::fmax(math::floor((w - wlo) / hw(iu, iv)), Real(0)), Real(nw - 1));
    return static_cast<int>(q);
  }
  int wIndexH(Real w, Real h) const
  {
    IFGF_NO_FP_CONTRACT
    const Real q = math::fmin(math::fmax(math::floor((w - wlo) / h), Real(0)), Real(nw - 1));
    return static_cast<int>(q);
  }
  std::int32_t key(int face, int iw, int iu, int iv) const { return ((face * nw + iw) * span + iu) * span + iv; }
  void decode(std::int32_t k, int& face, int& iw, int& iu, int& iv) const
  {
    iv = k % span, k /= span;
    iu = k % span, k /= span;
    iw = k % nw, face = k / nw;
  }
};

template <class Real>
struct FaceCoord {
  int face;
  Real w, u, v;
};

template <class Real>
inline Real faceComp(const Vec3<Real>& y, int a)
{
  return a == 0 ? y.x : (a == 1 ? y.y : y.z);
}

// coordinates of y in the chart of `face` (y.n must be > 0)
template <class Real>
inline FaceCoord<Real> inFace(const Vec3<Real>& y, Real H, int face)
{
  IFGF_NO_FP_CONTRACT
  const int a = face >> 1;
  const Real xn = (face & 1) ? -faceComp(y, a) : faceComp(y, a);
  const Real inv = Real(1) / xn;
  return {face, H * inv, faceComp(y, a == 2 ? 0 : a + 1) * inv, faceComp(y, a == 0 ? 2 : a - 1) * inv};
}

// natural face (largest |component|; ties to the lower axis) and its coordinates
template <class Real>
inline FaceCoord<Real> faceOf(const Vec3<Real>& y, Real H)
{
  const Real ax = math::fmax(y.x, -y.x), ay = math::fmax(y.y, -y.y), az = math::fmax(y.z, -y.z);
  const int a = (ax >= ay && ax >= az) ? 0 : (ay >= az ? 1 : 2);
  return inFace(y, H, 2 * a + (faceComp(y, a) < Real(0) ? 1 : 0));
}

// (selects instead of runtime-indexed arrays: those went to local memory on the device)
template <class Real>
inline Vec3<Real> fromFace(int face, Real w, Real u, Real v, Real H)
{
  IFGF_NO_FP_CONTRACT
  const int a = face >> 1;
  const Real xn = H / w, sx = (face & 1) ? -xn : xn, un = u * xn, vn = v * xn;
  // c[a] = sx, c[(a+1)%3] = un, c[(a+2)%3] = vn
  return {a == 0 ? sx : (a == 1 ? vn : un), a == 0 ? un : (a == 1 ? sx : vn), a == 0 ? vn : (a == 1 ? un : sx)};
}

// the projective map of one radial layer into the child of octant oct (child center at +-Hc/2 per axis)
template <class Real>
struct LayerMap {
  Real den, s1, s2, wc;
};
template <class Real>
inline LayerMap<Real> layerMap(int face, Real w, Real Hp, Real Hc, int oct)
{
  IFGF_NO_FP_CONTRACT
  const int a = face >> 1;
  const Real h = Hc * Real(0.5);
  const Real dx = (oct & 1) ? h : -h, dy = (oct & 2) ? h : -h, dz = (oct & 4) ? h : -h;
  const Real da = a == 0 ? dx : (a == 1 ? dy : dz);   // d[a]
  const Real db = a == 0 ? dy : (a == 1 ? dz : dx);   // d[(a+1)%3]
  const Real dc = a == 0 ? dz : (a == 1 ? dx : dy);   // d[(a+2)%3]
  const Real dn = (face & 1) ? -da : da;
  LayerMap<Real> m;
  m.den = Real(1) - dn * w / Hp;
  m.s1 = db * w / Hp;
  m.s2 = dc * w / Hp;
  m.wc = Hc / Hp * w / m.den;
  return m;
}

// element key of a natural-face point (primary chart) and its local coordinates
template <class Real>
inline std::int32_t locateFace(const FaceGrid<Real>& g, const FaceCoord<Real>& c, Real xi[3])
{
  const int iu = g.uIndexPrimary(c.u), iv = g.uIndexPrimary(c.v), iw = g.wIndex(c.w, iu, iv);
  xi[0] = g.wLocal(c.w, iw, iu, iv), xi[1] = g.uLocal(c.u, iu), xi[2] = g.uLocal(c.v, iv);
  return g.key(c.face, iw, iu, iv);
}

struct FaceParams {
  double alpha = 1.7320508075688772;
  double density = 0.0;  // elements per unit box side: C |kappa|
  double fw = 1.0, fu = 2.25;
  int nwMin = 2, nuMin = 4;
  double delta = 0.0;    // chart extension per side (u, v units; whole cells); 0: nodes beyond the face edge are
                         // evaluated as fallback lines in the neighbouring face (cheaper than extension cones)
  bool col = false;      // false: one w range [wlo, 1] for all columns (nested cube shells ||y||_inf = H / w);
                         // true: per-column range [wlo, smax sqrt(1 + U^2 + V^2)] (slower at equal accuracy)
  double marginUlps = 16.0;
};

// One structured child of a (parent element, octant): its element key and tasks [taskBegin, taskEnd).
// Task (packed): layer i << 16 | j0 << 12 | j1 << 8 | k0 << 4 | k1 (u-nodes [j0, j1), v-nodes [k0, k1)).
struct FaceChildRec {
  std::int32_t key, taskBegin, taskEnd;
};
// Fallback line: parent nodes of one radial layer that leave the child's chart and land in the same child cone
// `key` of the child's neighbouring face. Seen from the child, y = X - o has, in the parent's chart axes
// (a, b, c), components P_a (fixed per layer), P_b (affine in u_j) and P_c (affine in v_k). In the chart of
// face b the child coordinates are w' = Hc / |P_b|, u' = P_c / |P_b|, v' = P_a / |P_b|: for fixed (i, j) w' and
// v' are shared and u' varies with k. In the chart of face c, w' and u' are shared for fixed (i, k) and v'
// varies with j. So a line needs one radial and one angular contraction plus one short sum per node.
// axis: the shared angular axis (1: u', 2: v'); xiW, xiA: shared local coordinates; nodes [nBegin, nBegin + n).
struct FaceLine {
  std::int32_t key, axisN, nBegin;  // axisN = axis << 8 | n
  float xiW, xiA;
};
struct FaceLineNode {
  std::int32_t node;  // parent node index i + nW (j + nU k)
  float xi;           // the varying local coordinate
};

template <class Real>
struct FaceConeTables {
  interp::Orders orders;
  std::vector<FaceGrid<Real>> grid;  // per source level
  Vec3d c0{};                        // far-field reference point (source root center)
  std::vector<std::int32_t> coneOffset, coneEl;
  // per parent level l (children on l + 1): the active parent elements (sorted keys); the level maps themselves
  // stay on the device (FaceLevelMapDevice)
  struct LevelMap {
    std::vector<std::int32_t> U;
  };
  std::vector<LevelMap> maps;
  double farActivationMs = 0, parentActivationMs = 0, levelMapMs = 0;
  std::int32_t numCones() const { return static_cast<std::int32_t>(coneEl.size()); }
};

// Far-target activation of one target point (device).
// y = Real(t - c0) - Real(c_S - c0) as the far kernel computes it. Emits the natural face's element and the
// neighbours within the margin mu (relative) of every border, and the other face's elements when two components
// nearly tie. Keys may repeat.
template <class Real, class Emit>
IFGF_INLINE void faceFarKeys(const FaceGrid<Real>& g, const Vec3<Real>& y, Real H, double mu, Emit&& emit)
{
  IFGF_NO_FP_CONTRACT
  auto absd = [](Real x) { const double d = double(x); return d < 0 ? -d : d; };
  const FaceCoord<Real> nat = faceOf(y, H);
  const int a = nat.face >> 1;
  const double ca = absd(faceComp(y, a)) * (1 - mu);
  for (int fi = 0; fi < 3; ++fi) {
    FaceCoord<Real> fc = nat;
    if (fi > 0) {
      const int b = (a + fi) % 3;
      if (!(absd(faceComp(y, b)) >= ca)) continue;
      fc = inFace(y, H, 2 * b + (faceComp(y, b) < Real(0) ? 1 : 0));
    }
    const int iu0 = g.uIndexPrimary(fc.u), iv0 = g.uIndexPrimary(fc.v);
    const double du = mu * 2 / double(g.hu);
    const double fu = (double(fc.u) + 1) / double(g.hu) - (iu0 - g.ext);
    const double fv = (double(fc.v) + 1) / double(g.hu) - (iv0 - g.ext);
    const bool uLo = fu < du && iu0 > g.ext, uHi = fu > 1 - du && iu0 < g.ext + g.nu - 1;
    const bool vLo = fv < du && iv0 > g.ext, vHi = fv > 1 - du && iv0 < g.ext + g.nu - 1;
    for (int ou = -1; ou <= 1; ++ou) {
      if ((ou == -1 && !uLo) || (ou == 1 && !uHi)) continue;
      for (int ov = -1; ov <= 1; ++ov) {
        if ((ov == -1 && !vLo) || (ov == 1 && !vHi)) continue;
        const int iu = iu0 + ou, iv = iv0 + ov;
        const int iw0 = g.wIndex(fc.w, iu, iv);
        const double hw = double(g.hw(iu, iv)), xw = (double(fc.w) - double(g.wlo)) / hw;
        const double dw = mu / hw, fr = xw - iw0;
        emit(g.key(fc.face, iw0, iu, iv));
        if (fr < dw && iw0 > 0) emit(g.key(fc.face, iw0 - 1, iu, iv));
        if (fr > 1 - dw && iw0 < g.nw - 1) emit(g.key(fc.face, iw0 + 1, iu, iv));
      }
    }
  }
}

// Grids per source level and the reference point; coneOffset zeroed, maps sized. Returns false if there is
// nothing to build.
template <class Real>
bool initFaceGrids(FaceConeTables<Real>& c, const Tree& S, const Tree& T, const interp::Orders& orders, const FaceParams& p)
{
  if (!(p.alpha >= std::sqrt(3.0) / 2)) throw std::invalid_argument("buildFaceCones: alpha must be >= sqrt(3)/2");
  c.orders = orders;
  const int L = S.numLevels();
  c.coneOffset.assign(static_cast<std::size_t>(S.numBoxes()) + 1, 0);
  c.maps.resize(std::max(L, 1));
  if (L == 0 || T.numBoxes() == 0) return false;
  c.c0 = S.center[0];
  const double half3 = std::sqrt(3.0) / 2, smax = 1.0 / p.alpha;
  const Vec3d cS{S.rootMin.x + S.side / 2, S.rootMin.y + S.side / 2, S.rootMin.z + S.side / 2};
  const Vec3d cT{T.rootMin.x + T.side / 2, T.rootMin.y + T.side / 2, T.rootMin.z + T.side / 2};
  double R = norm(cS - cT) + half3 * (S.side + T.side);
  std::vector<double> wloD(L);
  for (int l = 0; l < L; ++l) {
    if (l > 0) R = S.H[l - 1] / wloD[l - 1] + half3 * S.H[l];
    wloD[l] = std::min(S.H[l] / R, 0.5 * smax);
    const int m = static_cast<int>(std::ceil(std::max(p.density * S.H[l], 1.0)));
    FaceGrid<Real> g;
    g.nw = std::max(p.nwMin, static_cast<int>(std::lround(p.fw * m)));
    g.nu = std::max(p.nuMin, static_cast<int>(std::lround(p.fu * m)));
    const double hu = 2.0 / g.nu;
    g.ext = static_cast<int>(std::ceil(p.delta / hu - 1e-12));
    g.span = g.nu + 2 * g.ext;
    g.col = p.col ? 1 : 0;
    g.wlo = static_cast<Real>(wloD[l]);
    g.hu = static_cast<Real>(hu);
    g.smax = static_cast<Real>(smax);
    g.hw0 = static_cast<Real>((1.0 - wloD[l]) / g.nw);
    if (6.0 * g.nw * double(g.span) * g.span >= double(std::numeric_limits<std::int32_t>::max()))
      throw std::runtime_error("buildFaceCones: element keys overflow int32");
    c.grid.push_back(g);
  }
  return true;
}

// ---- Level maps (box-independent, per parent level l, children on l + 1) ----
// Item = (active parent element U[u], octant): its structured children (child key, tasks: layer i, u-node range
// [j0, j1), v-node range [k0, k1)) and its fallback lines. One routine, faceLevelMapItem, for the host and the device
// builders (same arithmetic; the column widths hw come from host-computed tables, no sqrt on the device).

template <class Real>
struct FaceLevelMapCtx {
  FaceGrid<Real> gp, gc;  // parent and child level grids
  Real Hp, Hc;
  int nW, nU, nV;
  const Real* cheb;  // Chebyshev nodes: [0, nW) radial, [8, 8 + nU) u, [16, 16 + nV) v
  const Real* hwP;   // hw per parent column (iu * span + iv)
  const Real* hwC;   // hw per child column
};
struct FaceLevelMapCounts {
  std::int32_t children = 0, tasks = 0, lines = 0, lineNodes = 0;
  std::int64_t structuredNodes = 0;
};

// fallback node (layer i, u-node j, v-node k) of parent element (f, iw, iu, iv), octant offset o: child key and xi
template <class Real>
IFGF_INLINE std::int32_t faceFallbackNode(const FaceLevelMapCtx<Real>& x, int f, int iw, int iu, int iv, const Vec3<Real>& o,
                                          int i, int j, int k, Real xi[3], int& face)
{
  IFGF_NO_FP_CONTRACT
  const FaceGrid<Real>& gp = x.gp;
  const FaceGrid<Real>& gc = x.gc;
  const Real w = gp.wlo + (Real(iw) + Real(0.5) + Real(0.5) * x.cheb[i]) * x.hwP[iu * gp.span + iv];
  const Vec3<Real> X = fromFace(f, w, gp.uOf(iu, x.cheb[8 + j]), gp.uOf(iv, x.cheb[16 + k]), x.Hp);
  const FaceCoord<Real> fc = faceOf(X - o, x.Hc);
  const int ju = gc.uIndexPrimary(fc.u), jv = gc.uIndexPrimary(fc.v);
  const Real hh = x.hwC[ju * gc.span + jv];
  const int jw = gc.wIndexH(fc.w, hh);
  xi[0] = gc.wLocalH(fc.w, jw, hh), xi[1] = gc.uLocal(fc.u, ju), xi[2] = gc.uLocal(fc.v, jv);
  face = fc.face;
  return gc.key(fc.face, jw, ju, jv);
}

// One item. Scratch kt and fb: nW nU nV entries each (at most 512). Fill = false: counts only. Fill = true: writes
// children (taskBegin/End relative to taskBase), tasks, lines (nBegin relative to nodeBase) and line nodes.
// Order: children by (key, task); lines by (key, layer, axis, line index, node) - as one packed 64-bit integer
// (key 31 | layer 3 | axis 2 | line index 10 | node 9 bits); nodes of a line split if their float xi disagree.
template <bool Fill, class Real>
IFGF_INLINE FaceLevelMapCounts faceLevelMapItem(const FaceLevelMapCtx<Real>& x, std::int32_t e, int oct, std::uint64_t* kt,
                                                std::uint64_t* fb, FaceChildRec* chOut, std::uint32_t* tkOut, FaceLine* lnOut,
                                                FaceLineNode* lnnOut, std::int32_t taskBase, std::int32_t nodeBase)
{
  IFGF_NO_FP_CONTRACT
  const FaceGrid<Real>& gp = x.gp;
  const FaceGrid<Real>& gc = x.gc;
  FaceLevelMapCounts cnt;
  int f, iw, iu, iv;
  gp.decode(e, f, iw, iu, iv);
  const int pa = f >> 1, pb = (pa + 1) % 3;
  const Real h = x.Hc * Real(0.5);
  const Vec3<Real> o{(oct & 1) ? h : -h, (oct & 2) ? h : -h, (oct & 4) ? h : -h};
  const Real hwp = x.hwP[iu * gp.span + iv];
  int nkt = 0, nfb = 0;
  for (int i = 0; i < x.nW; ++i) {
    const Real w = gp.wlo + (Real(iw) + Real(0.5) + Real(0.5) * x.cheb[i]) * hwp;
    const LayerMap<Real> m = layerMap(f, w, x.Hp, x.Hc, oct);
    int cu[8], cv[8];
    bool okU[8], okV[8];
    for (int j = 0; j < x.nU; ++j) {
      cu[j] = gc.uIndex((gp.uOf(iu, x.cheb[8 + j]) - m.s1) / m.den);
      okU[j] = m.den > Real(0) && cu[j] >= 0 && cu[j] < gc.span;
    }
    for (int k = 0; k < x.nV; ++k) {
      cv[k] = gc.uIndex((gp.uOf(iv, x.cheb[16 + k]) - m.s2) / m.den);
      okV[k] = m.den > Real(0) && cv[k] >= 0 && cv[k] < gc.span;
    }
    // runs of consecutive structured nodes with the same child index (u_c is monotonic in j)
    for (int j0 = 0; j0 < x.nU;) {
      int j1 = j0 + 1;
      while (j1 < x.nU && okU[j1] == okU[j0] && cu[j1] == cu[j0]) ++j1;
      if (okU[j0])
        for (int k0 = 0; k0 < x.nV;) {
          int k1 = k0 + 1;
          while (k1 < x.nV && okV[k1] == okV[k0] && cv[k1] == cv[k0]) ++k1;
          if (okV[k0]) {
            const std::int32_t key = gc.key(f, gc.wIndexH(m.wc, x.hwC[cu[j0] * gc.span + cv[k0]]), cu[j0], cv[k0]);
            const std::uint32_t task = static_cast<std::uint32_t>(i << 16 | j0 << 12 | j1 << 8 | k0 << 4 | k1);
            kt[nkt++] = std::uint64_t(static_cast<std::uint32_t>(key)) << 32 | task;
            cnt.structuredNodes += (j1 - j0) * (k1 - k0);
          }
          k0 = k1;
        }
      j0 = j1;
    }
    for (int j = 0; j < x.nU; ++j)
      for (int k = 0; k < x.nV; ++k) {
        if (okU[j] && okV[k]) continue;
        Real xi[3];
        int face;
        const std::int32_t key = faceFallbackNode(x, f, iw, iu, iv, o, i, j, k, xi, face);
        const int node = i + x.nW * (j + x.nU * k), fa = face >> 1;
        // face b: fixed (i, j), shared v' (axis 2); face c: fixed (i, k), shared u' (axis 1); otherwise own line
        const int axis = fa == pb ? 2 : 1, lineIdx = fa == pb ? j : (fa != pa ? k : 16 + node);
        fb[nfb++] = std::uint64_t(static_cast<std::uint32_t>(key)) << 24 | std::uint64_t(i) << 21 | std::uint64_t(axis) << 19 |
                    std::uint64_t(lineIdx) << 9 | std::uint64_t(node);
      }
  }
  // insertion sorts (short lists; all entries distinct)
  auto isort = [](std::uint64_t* a, int n) {
    for (int p = 1; p < n; ++p) {
      const std::uint64_t v = a[p];
      int q = p - 1;
      for (; q >= 0 && a[q] > v; --q) a[q + 1] = a[q];
      a[q + 1] = v;
    }
  };
  isort(kt, nkt);
  isort(fb, nfb);
  // structured children
  for (int a = 0; a < nkt;) {
    const std::uint32_t key = static_cast<std::uint32_t>(kt[a] >> 32);
    const std::int32_t tb = cnt.tasks;
    int b = a;
    for (; b < nkt && static_cast<std::uint32_t>(kt[b] >> 32) == key; ++b) {
      if constexpr (Fill) tkOut[cnt.tasks] = static_cast<std::uint32_t>(kt[b]);
      ++cnt.tasks;
    }
    if constexpr (Fill) chOut[cnt.children] = {static_cast<std::int32_t>(key), taskBase + tb, taskBase + cnt.tasks};
    ++cnt.children;
    a = b;
  }
  // fallback lines (xi recomputed per node: same function, same values)
  auto xiOf = [&](std::uint64_t ord, Real xi[3]) {
    const int node = static_cast<int>(ord & 511), i = node % x.nW, j = (node / x.nW) % x.nU, k = node / (x.nW * x.nU);
    int face;
    faceFallbackNode(x, f, iw, iu, iv, o, i, j, k, xi, face);
  };
  for (int q0 = 0; q0 < nfb;) {
    const std::uint64_t hd = fb[q0];
    const std::uint64_t grp = hd >> 9;  // key, layer, axis, line index
    const int A = static_cast<int>((hd >> 19) & 3), V = 3 - A;
    Real hx[3];
    xiOf(hd, hx);
    const float h0 = static_cast<float>(hx[0]), hA = static_cast<float>(hx[A]);
    const std::int32_t nb = cnt.lineNodes;
    int q1 = q0;
    for (; q1 < nfb; ++q1) {
      Real qx[3];
      if (q1 == q0) {
        qx[0] = hx[0], qx[1] = hx[1], qx[2] = hx[2];
      } else {
        if ((fb[q1] >> 9) != grp) break;
        xiOf(fb[q1], qx);
        const float d0 = static_cast<float>(qx[0]) - h0, dA = static_cast<float>(qx[A]) - hA;
        if (!((d0 < 0 ? -d0 : d0) < 1e-5f && (dA < 0 ? -dA : dA) < 1e-5f)) break;
      }
      if constexpr (Fill) lnnOut[cnt.lineNodes] = {static_cast<std::int32_t>(fb[q1] & 511), static_cast<float>(qx[V])};
      ++cnt.lineNodes;
    }
    if constexpr (Fill) {
      FaceLine ln;
      ln.key = static_cast<std::int32_t>(hd >> 24);
      ln.axisN = A << 8 | (q1 - q0);
      ln.nBegin = nodeBase + nb;
      ln.xiW = h0, ln.xiA = hA;
      lnOut[cnt.lines] = ln;
    }
    ++cnt.lines;
    q0 = q1;
  }
  return cnt;
}

// context tables for parent level l: Chebyshev nodes (3 x 8), hw per parent and per child column
template <class Real>
void faceLevelMapTables(const FaceConeTables<Real>& c, int l, std::vector<Real>& cheb, std::vector<Real>& hwP, std::vector<Real>& hwC)
{
  const auto& o = c.orders;
  if (o.n[0] > 8 || o.n[1] > 8 || o.n[2] > 8 || o.nodes() > 512)
    throw std::invalid_argument("faceLevelMap: orders exceed the packed line order (n <= 8, nodes <= 512)");
  cheb.assign(24, Real(0));
  for (int d = 0; d < 3; ++d)
    for (int i = 0; i < o.n[d]; ++i) cheb[8 * d + i] = static_cast<Real>(interp::chebNode(i, o.n[d]));
  auto table = [](const FaceGrid<Real>& g, std::vector<Real>& t) {
    t.resize(static_cast<std::size_t>(g.span) * g.span);
    for (int a = 0; a < g.span; ++a)
      for (int b = 0; b < g.span; ++b) t[a * g.span + b] = g.hw(a, b);
  };
  table(c.grid[l], hwP);
  table(c.grid[l + 1], hwC);
}

template <class Real>
FaceLevelMapCtx<Real> faceLevelMapCtx(const FaceConeTables<Real>& c, const Tree& S, int l, const std::vector<Real>& cheb,
                                      const std::vector<Real>& hwP, const std::vector<Real>& hwC)
{
  return {c.grid[l],     c.grid[l + 1],  static_cast<Real>(S.H[l]), static_cast<Real>(S.H[l + 1]), c.orders.n[0],
          c.orders.n[1], c.orders.n[2],  cheb.data(),               hwP.data(),                    hwC.data()};
}

}  // namespace ifgf::cones
