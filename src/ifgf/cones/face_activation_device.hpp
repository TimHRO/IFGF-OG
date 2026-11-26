#pragma once

// Face-chart cone activation on the device:
//   - one bitmap per source box over its level's element keys (6 nw span^2 bits; keys are bit positions, so the
//     extracted cone lists are sorted for free)
//   - far targets: one work-group per far pair, one work item per target point, faceFarKeys -> atomic OR
//   - per parent level, top-down: union of the parents' bitmaps -> active parent elements U (host extract), the
//     box-independent level map on the device (buildFaceLevelMapDevice: faceLevelMapItem per
//     (element, octant) work item; count, host scan, fill), then child activation on the device: per (parent box,
//     active element, octant) OR the child keys of the level map into the child's bitmap
//   - count + host scan + extract -> coneOffset, coneEl (downloaded; small)
// The level maps stay on the device (FaceConesDevice::maps; the upward plan builder reads them there). They are
// built for a given set of active parent elements U, so a later multi-device version can build each device's part.

#include <ifgf/cones/face_cones.hpp>
#include <ifgf/exec/device_buffer.hpp>

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdint>
#include <vector>

namespace ifgf::cones {

// level map of one parent level on the device (same layout as FaceConeTables::LevelMap, plus the dense key -> U
// index map of the parent level)
struct FaceLevelMapDevice {
  std::size_t numItems = 0, numChildren = 0, numTasks = 0, numLines = 0, numLineNodes = 0;
  exec::DeviceBuffer<std::int32_t> U, uIdx, childOff, lineOff;
  exec::DeviceBuffer<FaceChildRec> child;
  exec::DeviceBuffer<std::uint32_t> task;
  exec::DeviceBuffer<FaceLine> line;
  exec::DeviceBuffer<FaceLineNode> lineNode;
};

template <class Real>
struct FaceConesDevice {
  FaceConeTables<Real> tables;           // cones; maps[l].U (active parent elements)
  std::vector<FaceLevelMapDevice> maps;  // per parent level
};

// Level map of parent level l for the active parent elements c.maps[l].U (sorted) on the device. Each item is
// computed ONCE (faceLevelMapItem<true>) into its own worst-case slots of a per-batch scratch (every count <= nodes),
// then compacted: counts and content come from the same evaluation. (A count pass + fill pass would have to decide
// every tie identically twice: parent nodes exactly on a child face or cell border go to either side depending on
// the last bit, which differs between builds - fine for one consistent map, fatal for count/fill.)
template <class Real>
FaceLevelMapDevice buildFaceLevelMapDevice(sycl::queue& q, FaceConeTables<Real>& c, const Tree& S, int l)
{
  using exec::DeviceBuffer;
  FaceLevelMapDevice D;
  const auto& U = c.maps[l].U;
  const std::size_t nItems = U.size() * 8;
  D.numItems = nItems;
  const auto& gp = c.grid[l];
  {
    std::vector<std::int32_t> uIdx(static_cast<std::size_t>(6) * gp.nw * gp.span * gp.span, -1);
    for (std::size_t u = 0; u < U.size(); ++u) uIdx[U[u]] = static_cast<std::int32_t>(u);
    D.uIdx = DeviceBuffer<std::int32_t>(q, uIdx);
  }
  D.U = DeviceBuffer<std::int32_t>(q, U.empty() ? std::vector<std::int32_t>{0} : U);
  if (nItems == 0) {
    D.childOff = DeviceBuffer<std::int32_t>(q, std::vector<std::int32_t>{0});
    D.lineOff = DeviceBuffer<std::int32_t>(q, std::vector<std::int32_t>{0});
    D.child = DeviceBuffer<FaceChildRec>(q, 1), D.task = DeviceBuffer<std::uint32_t>(q, 1);
    D.line = DeviceBuffer<FaceLine>(q, 1), D.lineNode = DeviceBuffer<FaceLineNode>(q, 1);
    return D;
  }
  std::vector<Real> cheb, hwP, hwC;
  faceLevelMapTables(c, l, cheb, hwP, hwC);
  DeviceBuffer<Real> dCheb(q, cheb), dHwP(q, hwP), dHwC(q, hwC);
  FaceLevelMapCtx<Real> x = faceLevelMapCtx(c, S, l, cheb, hwP, hwC);
  x.cheb = dCheb.data(), x.hwP = dHwP.data(), x.hwC = dHwC.data();
  const std::size_t nodes = static_cast<std::size_t>(c.orders.nodes());
  constexpr std::size_t batch = std::size_t(1) << 14;
  const std::size_t nb0 = std::min(batch, nItems);
  // per-item worst-case slots: sort scratch (2 nodes), children, tasks, lines, line nodes (nodes each)
  DeviceBuffer<std::uint64_t> sKt(q, nb0 * 2 * nodes);
  DeviceBuffer<FaceChildRec> sCh(q, nb0 * nodes);
  DeviceBuffer<std::uint32_t> sTk(q, nb0 * nodes);
  DeviceBuffer<FaceLine> sLn(q, nb0 * nodes);
  DeviceBuffer<FaceLineNode> sLnn(q, nb0 * nodes);
  DeviceBuffer<FaceLevelMapCounts> cnt(q, nb0);
  DeviceBuffer<std::int64_t> bOff(q, nb0 * 4);  // per batch item: child, task, line, node offset in the level
  struct Chunk {
    std::size_t ch, tk, ln, lnn;  // sizes
    DeviceBuffer<FaceChildRec> child;
    DeviceBuffer<std::uint32_t> task;
    DeviceBuffer<FaceLine> line;
    DeviceBuffer<FaceLineNode> lineNode;
  };
  std::vector<Chunk> chunks;
  std::vector<std::int32_t> childOff(nItems + 1, 0), lineOff(nItems + 1, 0);
  std::int64_t tkRun = 0, ndRun = 0;
  const std::int32_t* up = D.U.data();
  std::uint64_t* sk = sKt.data();
  FaceChildRec* sc = sCh.data();
  std::uint32_t* st = sTk.data();
  FaceLine* sl = sLn.data();
  FaceLineNode* sn = sLnn.data();
  FaceLevelMapCounts* cp = cnt.data();
  std::int64_t* bo = bOff.data();
  for (std::size_t b0 = 0; b0 < nItems; b0 += batch) {
    const std::size_t nb = std::min(batch, nItems - b0);
    q.parallel_for(sycl::range<1>{nb}, [=](sycl::id<1> id) {
       const std::size_t i = id[0], idx = b0 + i;
       std::uint64_t* kt = sk + i * 2 * nodes;
       cp[i] = faceLevelMapItem<true>(x, up[idx / 8], static_cast<int>(idx % 8), kt, kt + nodes, sc + i * nodes, st + i * nodes,
                                      sl + i * nodes, sn + i * nodes, 0, 0);
     }).wait_and_throw();
    const auto cv = cnt.download();
    std::vector<std::int64_t> off(nb * 4);
    std::int64_t ch = 0, tk = 0, ln = 0, nd = 0;
    for (std::size_t i = 0; i < nb; ++i) {
      off[4 * i] = ch, off[4 * i + 1] = tk, off[4 * i + 2] = ln, off[4 * i + 3] = nd;
      ch += cv[i].children, tk += cv[i].tasks, ln += cv[i].lines, nd += cv[i].lineNodes;
      childOff[b0 + i + 1] = childOff[b0 + i] + cv[i].children;
      lineOff[b0 + i + 1] = lineOff[b0 + i] + cv[i].lines;
    }
    q.copy(off.data(), bo, off.size()).wait();
    Chunk k{static_cast<std::size_t>(ch), static_cast<std::size_t>(tk), static_cast<std::size_t>(ln), static_cast<std::size_t>(nd),
            DeviceBuffer<FaceChildRec>(q, std::max<std::int64_t>(ch, 1)), DeviceBuffer<std::uint32_t>(q, std::max<std::int64_t>(tk, 1)),
            DeviceBuffer<FaceLine>(q, std::max<std::int64_t>(ln, 1)), DeviceBuffer<FaceLineNode>(q, std::max<std::int64_t>(nd, 1))};
    FaceChildRec* kc = k.child.data();
    std::uint32_t* kk = k.task.data();
    FaceLine* kl = k.line.data();
    FaceLineNode* kn = k.lineNode.data();
    const std::int32_t tb0 = static_cast<std::int32_t>(tkRun), nb0r = static_cast<std::int32_t>(ndRun);
    q.parallel_for(sycl::range<1>{nb}, [=](sycl::id<1> id) {
       const std::size_t i = id[0];
       const FaceLevelMapCounts n = cp[i];
       const std::int64_t* o = bo + 4 * i;
       const std::int32_t tb = tb0 + static_cast<std::int32_t>(o[1]), nbase = nb0r + static_cast<std::int32_t>(o[3]);
       for (std::int32_t r = 0; r < n.children; ++r) {
         FaceChildRec cr = sc[i * nodes + r];
         cr.taskBegin += tb, cr.taskEnd += tb;
         kc[o[0] + r] = cr;
       }
       for (std::int32_t r = 0; r < n.tasks; ++r) kk[o[1] + r] = st[i * nodes + r];
       for (std::int32_t r = 0; r < n.lines; ++r) {
         FaceLine ln = sl[i * nodes + r];
         ln.nBegin += nbase;
         kl[o[2] + r] = ln;
       }
       for (std::int32_t r = 0; r < n.lineNodes; ++r) kn[o[3] + r] = sn[i * nodes + r];
     }).wait_and_throw();
    tkRun += tk, ndRun += nd;
    if (tkRun >= std::numeric_limits<std::int32_t>::max() || ndRun >= std::numeric_limits<std::int32_t>::max())
      throw std::runtime_error("buildFaceLevelMapDevice: level map overflows int32");
    chunks.push_back(std::move(k));
  }
  D.numChildren = childOff[nItems], D.numTasks = static_cast<std::size_t>(tkRun), D.numLines = lineOff[nItems],
  D.numLineNodes = static_cast<std::size_t>(ndRun);
  D.childOff = DeviceBuffer<std::int32_t>(q, childOff);
  D.lineOff = DeviceBuffer<std::int32_t>(q, lineOff);
  D.child = DeviceBuffer<FaceChildRec>(q, std::max<std::size_t>(D.numChildren, 1));
  D.task = DeviceBuffer<std::uint32_t>(q, std::max<std::size_t>(D.numTasks, 1));
  D.line = DeviceBuffer<FaceLine>(q, std::max<std::size_t>(D.numLines, 1));
  D.lineNode = DeviceBuffer<FaceLineNode>(q, std::max<std::size_t>(D.numLineNodes, 1));
  std::size_t pc = 0, pt = 0, pl = 0, pn = 0;
  for (const auto& k : chunks) {
    if (k.ch) q.copy(k.child.data(), D.child.data() + pc, k.ch);
    if (k.tk) q.copy(k.task.data(), D.task.data() + pt, k.tk);
    if (k.ln) q.copy(k.line.data(), D.line.data() + pl, k.ln);
    if (k.lnn) q.copy(k.lineNode.data(), D.lineNode.data() + pn, k.lnn);
    pc += k.ch, pt += k.tk, pl += k.ln, pn += k.lnn;
  }
  q.wait_and_throw();
  return D;
}

template <class Real>
FaceConesDevice<Real> buildFaceConesDevice(sycl::queue& q, const Tree& S, const Tree& T, const InteractionLists& lists,
                                           const interp::Orders& orders, const FaceParams& p)
{
  using exec::DeviceBuffer;
  using Word = std::uint32_t;
  using Atom = sycl::atomic_ref<Word, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                sycl::access::address_space::global_space>;
  constexpr int G = 128;
  FaceConesDevice<Real> out;
  FaceConeTables<Real>& c = out.tables;
  if (!initFaceGrids(c, S, T, orders, p)) return out;
  out.maps.resize(c.maps.size());
  const auto tA = std::chrono::steady_clock::now();
  const int L = S.numLevels();
  const std::int32_t NB = S.numBoxes();
  const double mu = p.marginUlps * std::numeric_limits<Real>::epsilon();

  // bitmap layout: per box the words of its level
  std::vector<std::int64_t> lvlWords(L), bitOff(static_cast<std::size_t>(NB) + 1, 0);
  for (int l = 0; l < L; ++l) {
    const auto& g = c.grid[l];
    lvlWords[l] = (6LL * g.nw * g.span * g.span + 31) / 32;
  }
  for (std::int32_t b = 0; b < NB; ++b) bitOff[b + 1] = bitOff[b] + lvlWords[S.level[b]];
  DeviceBuffer<Word> bits(q, static_cast<std::size_t>(bitOff[NB]));
  q.fill(bits.data(), Word(0), bits.size());
  DeviceBuffer<std::int64_t> dOff(q, bitOff);
  Word* bp = bits.data();
  const std::int64_t* op = dOff.data();

  // (a) far targets
  {
    const std::size_t np = lists.far.src.size();
    std::vector<std::int32_t> pT(np);
    for (std::int32_t t = 0; t < T.numBoxes(); ++t)
      for (auto k = lists.far.offsets[t]; k < lists.far.offsets[t + 1]; ++k) pT[k] = t;
    std::vector<Vec3<Real>> cen(NB), tp(T.numPoints());
    std::vector<std::int32_t> lev(NB);
    for (std::int32_t b = 0; b < NB; ++b) cen[b] = vec3Cast<Real>(S.center[b] - c.c0), lev[b] = S.level[b];
    for (std::uint32_t i = 0; i < T.numPoints(); ++i) tp[i] = vec3Cast<Real>(T.points[i] - c.c0);
    std::vector<Real> Hl(L);
    for (int l = 0; l < L; ++l) Hl[l] = static_cast<Real>(S.H[l]);
    DeviceBuffer<std::int32_t> dPT(q, pT), dPS(q, lists.far.src), dLev(q, lev);
    DeviceBuffer<Vec3<Real>> dCen(q, cen), dTp(q, tp);
    DeviceBuffer<Real> dH(q, Hl);
    DeviceBuffer<FaceGrid<Real>> dGrid(q, c.grid);
    DeviceBuffer<std::uint32_t> dTb(q, T.ptBegin), dTe(q, T.ptEnd);
    if (np > 0) {
      const std::int32_t *pt = dPT.data(), *ps = dPS.data(), *lv = dLev.data();
      const Vec3<Real>*ce = dCen.data(), *tq = dTp.data();
      const Real* hp = dH.data();
      const FaceGrid<Real>* gr = dGrid.data();
      const std::uint32_t *tb = dTb.data(), *te = dTe.data();
      q.parallel_for(sycl::nd_range<1>{np * G, G}, [=](sycl::nd_item<1> it) {
         const std::size_t n = it.get_group(0);
         const std::int32_t s = ps[n], t = pt[n], l = lv[s];
         const FaceGrid<Real> g = gr[l];
         const Real H = hp[l];
         const Vec3<Real> ctr = ce[s];
         Word* base = bp + op[s];
         auto emit = [&](std::int32_t key) {
           Atom(base[key >> 5]).fetch_or(Word(1) << (key & 31));
         };
         for (std::uint32_t i = tb[t] + static_cast<std::uint32_t>(it.get_local_id(0)); i < te[t]; i += G)
           faceFarKeys(g, tq[i] - ctr, H, mu, emit);
       }).wait_and_throw();
    }
  }
  const auto tB = std::chrono::steady_clock::now();
  c.farActivationMs = std::chrono::duration<double, std::milli>(tB - tA).count();

  // (b) parent levels, top-down
  std::vector<std::int32_t> childFlat(static_cast<std::size_t>(NB) * 8);
  for (std::int32_t b = 0; b < NB; ++b)
    for (int o = 0; o < 8; ++o) childFlat[b * 8 + o] = S.child[b][o];
  DeviceBuffer<std::int32_t> dChild(q, childFlat);
  const std::int32_t* chp = dChild.data();
  for (int l = 1; l < L; ++l) {
    const int pl = l - 1;
    auto& M = c.maps[pl];
    std::vector<std::int32_t> pbs;
    for (auto pb = S.levelBegin[pl]; pb < S.levelBegin[l]; ++pb)
      if (!S.isLeaf[pb]) pbs.push_back(pb);
    const std::int64_t W = lvlWords[pl];
    if (!pbs.empty()) {
      DeviceBuffer<std::int32_t> dPbs(q, pbs);
      DeviceBuffer<Word> lb(q, static_cast<std::size_t>(W));
      q.fill(lb.data(), Word(0), lb.size());
      const std::int32_t* pp = dPbs.data();
      Word* lbp = lb.data();
      q.parallel_for(sycl::range<1>{pbs.size() * static_cast<std::size_t>(W)}, [=](sycl::id<1> id) {
         const std::size_t k = id[0], w = k % W;
         const Word v = bp[op[pp[k / W]] + w];
         if (v) Atom(lbp[w]).fetch_or(v);
       }).wait_and_throw();
      const auto u = lb.download();
      for (std::int64_t w = 0; w < W; ++w)
        for (Word v = u[w]; v; v &= v - 1) M.U.push_back(static_cast<std::int32_t>(w * 32 + __builtin_ctz(v)));
    }
    {
      const auto tM = std::chrono::steady_clock::now();
      out.maps[pl] = buildFaceLevelMapDevice(q, c, S, pl);
      c.levelMapMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tM).count();
    }
    if (pbs.empty() || M.U.empty()) continue;
    const FaceLevelMapDevice& D = out.maps[pl];
    DeviceBuffer<std::int32_t> dPbs(q, pbs);
    const std::int32_t *pp = dPbs.data(), *ui = D.uIdx.data(), *cho = D.childOff.data(), *lno = D.lineOff.data();
    const FaceChildRec* chr = D.child.data();
    const FaceLine* lnr = D.line.data();
    q.parallel_for(sycl::nd_range<1>{pbs.size() * G, G}, [=](sycl::nd_item<1> it) {
       const std::int32_t pb = pp[it.get_group(0)];
       const Word* pw = bp + op[pb];
       for (std::int64_t w = static_cast<std::int64_t>(it.get_local_id(0)); w < W; w += G)
         for (Word v = pw[w]; v; v &= v - 1) {
           const std::int32_t e = static_cast<std::int32_t>(w * 32 + sycl::ctz(v));
           const std::int32_t u = ui[e];
           for (int oct = 0; oct < 8; ++oct) {
             const std::int32_t cb = chp[pb * 8 + oct];
             if (cb < 0) continue;
             Word* cw = bp + op[cb];
             for (auto r = cho[u * 8 + oct]; r < cho[u * 8 + oct + 1]; ++r) {
               const std::int32_t key = chr[r].key;
               Atom(cw[key >> 5]).fetch_or(Word(1) << (key & 31));
             }
             for (auto r = lno[u * 8 + oct]; r < lno[u * 8 + oct + 1]; ++r) {
               const std::int32_t key = lnr[r].key;
               Atom(cw[key >> 5]).fetch_or(Word(1) << (key & 31));
             }
           }
         }
     }).wait_and_throw();
  }

  // (c) count, scan (host), extract
  {
    DeviceBuffer<std::int32_t> cnt(q, static_cast<std::size_t>(NB));
    q.fill(cnt.data(), 0, cnt.size());
    std::int32_t* cp = cnt.data();
    auto chunk = [=](std::int32_t b, int lid, std::int64_t& w0, std::int64_t& w1) {
      const std::int64_t W = op[b + 1] - op[b], len = (W + G - 1) / G;
      w0 = op[b] + std::min<std::int64_t>(W, lid * len);
      w1 = op[b] + std::min<std::int64_t>(W, (lid + 1) * len);
    };
    q.parallel_for(sycl::nd_range<1>{static_cast<std::size_t>(NB) * G, G}, [=](sycl::nd_item<1> it) {
       const std::int32_t b = static_cast<std::int32_t>(it.get_group(0));
       std::int64_t w0, w1;
       chunk(b, static_cast<int>(it.get_local_id(0)), w0, w1);
       std::int32_t n = 0;
       for (auto w = w0; w < w1; ++w) n += static_cast<std::int32_t>(sycl::popcount(bp[w]));
       if (n)
         sycl::atomic_ref<std::int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                          sycl::access::address_space::global_space>(cp[b])
             .fetch_add(n);
     }).wait_and_throw();
    const auto cv = cnt.download();
    for (std::int32_t b = 0; b < NB; ++b) c.coneOffset[b + 1] = c.coneOffset[b] + cv[b];
    DeviceBuffer<std::int32_t> dCo(q, c.coneOffset), dEl(q, static_cast<std::size_t>(c.coneOffset[NB]));
    const std::int32_t* co = dCo.data();
    std::int32_t* el = dEl.data();
    if (c.coneOffset[NB] > 0)
      q.submit([&](sycl::handler& h) {
         sycl::local_accessor<std::int32_t, 1> lc{sycl::range<1>{G}, h};
         h.parallel_for(sycl::nd_range<1>{static_cast<std::size_t>(NB) * G, G}, [=](sycl::nd_item<1> it) {
           const std::int32_t b = static_cast<std::int32_t>(it.get_group(0));
           const int lid = static_cast<int>(it.get_local_id(0));
           std::int64_t w0, w1;
           chunk(b, lid, w0, w1);
           std::int32_t n = 0;
           for (auto w = w0; w < w1; ++w) n += static_cast<std::int32_t>(sycl::popcount(bp[w]));
           lc[lid] = n;
           sycl::group_barrier(it.get_group());
           if (lid == 0) {
             std::int32_t s = 0;
             for (int k = 0; k < G; ++k) {
               const std::int32_t x = lc[k];
               lc[k] = s;
               s += x;
             }
           }
           sycl::group_barrier(it.get_group());
           std::int32_t pos = co[b] + lc[lid];
           for (auto w = w0; w < w1; ++w)
             for (Word v = bp[w]; v; v &= v - 1)
               el[pos++] = static_cast<std::int32_t>((w - op[b]) * 32 + sycl::ctz(v));
         });
       }).wait_and_throw();
    c.coneEl = dEl.download();
    c.coneEl.resize(static_cast<std::size_t>(c.coneOffset[NB]));
  }
  c.parentActivationMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tB).count();
  return out;
}

}  // namespace ifgf::cones
