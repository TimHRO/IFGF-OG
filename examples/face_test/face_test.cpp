// face_test: one IFGF operator (FacePlan) on N random points on the unit sphere, targets = sources, radial normals,
// random real weights in [-1, 1]. Prints the init and per-stage apply times and the relative l2 error against the
// FP64 direct sum at 200 random targets.
//
// usage: face_test [N=100000] [kappa.re=12.56] [kappa.im=0.001] [SL|DL|CF] [leafSize=64] [C=0.15] [delta=0]
//                  [reps=3] [fu=2.25] [fw=1] [P=8]
//   kappa      wavenumber kappa.re + i kappa.im (the sphere has diameter 2, so kappa D = 2 kappa.re)
//   SL|DL|CF   single layer, double layer, combined field (eta = kappa)
//   leafSize   maximum points per leaf box
//   C          cone density: face elements per unit box side and unit wavenumber
//   delta      face-chart extension (0: fallback lines)
//   reps       timed applies after the first one
//   fu, fw     angular and radial grid factors per face
//   P          angular interpolation order 4..8 (radial order P - 2)
// IFGF_DEVICE = gpu (default; the CPU if there is no GPU) | cpu | <device index>

#include <ifgf/exec/device.hpp>
#include <ifgf/kernels/direct_sum.hpp>
#include <ifgf/plan/face_plan.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace ifgf;
using kernels::KappaKind;
using cd = std::complex<double>;

namespace {

// seeded uniform numbers in [0, 1) (the transform is written out, so the point cloud does not depend on the
// standard library's distribution implementation)
class Rng {
public:
  explicit Rng(std::uint64_t seed) : gen_(seed) {}
  double uniform() { return static_cast<double>(gen_() >> 11) * 0x1p-53; }
  double uniform(double a, double b) { return a + (b - a) * uniform(); }

private:
  std::mt19937_64 gen_;
};

std::vector<Vec3d> sphereSurface(std::size_t n, std::uint64_t seed)
{
  Rng rng(seed);
  std::vector<Vec3d> p(n);
  for (auto& x : p) {
    const double z = rng.uniform(-1.0, 1.0), phi = rng.uniform(0.0, 2.0 * M_PI);
    const double rho = std::sqrt(1.0 - z * z);
    x = Vec3d{rho * std::cos(phi), rho * std::sin(phi), z};
  }
  return p;
}

sycl::device pickDevice()
{
  const char* v = std::getenv("IFGF_DEVICE");
  if (v && *v) return exec::selectDevice(v);
  for (const auto& d : sycl::device::get_devices())
    if (d.is_gpu()) return d;
  return sycl::device{sycl::cpu_selector_v};
}

template <class KF, class KD, int P>
void run(sycl::queue& q, const std::vector<Vec3d>& p, const std::vector<Vec3d>& n, cd kappa, const FacePlanOptions& opt, int reps)
{
  const auto t0 = std::chrono::steady_clock::now();
  FacePlan<KF, P> plan(q, p, p, kappa, opt, n);
  const double initS = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  Rng rng(1);
  std::vector<typename KF::Value> w(p.size());
  std::vector<typename KD::Value> wd(p.size());
  for (std::size_t j = 0; j < p.size(); ++j) {
    const float v = static_cast<float>(rng.uniform(-1, 1));
    w[j] = v;
    wd[j] = v;
  }
  auto u = plan.apply(w);
  const double firstMs = plan.stats().applyMs;
  double best = 1e300, sum = 0;
  FacePlanStats st = plan.stats();
  for (int r = 0; r < reps; ++r) {
    u = plan.apply(w);
    sum += plan.stats().applyMs;
    if (plan.stats().applyMs < best) best = plan.stats().applyMs, st = plan.stats();
  }
  if (reps == 0) best = sum = firstMs;

  // relative l2 error at 200 random targets against the FP64 direct sum
  std::vector<std::size_t> idx(200);
  std::vector<Vec3d> ts(idx.size());
  for (std::size_t i = 0; i < idx.size(); ++i) {
    idx[i] = static_cast<std::size_t>(rng.uniform() * p.size());
    ts[i] = p[idx[i]];
  }
  const KD kd(kappa, opt.selfTolRel * plan.sourceTree().side);
  std::vector<typename KD::SrcData> sd(p.size());
  if constexpr (!std::is_empty_v<typename KD::SrcData>)
    for (std::size_t j = 0; j < p.size(); ++j) sd[j].n = n[j];
  const auto ref = kernels::directSum(kd, std::span<const Vec3d>(p), std::span<const typename KD::SrcData>(sd),
                                      std::span<const typename KD::Value>(wd), std::span<const Vec3d>(ts));
  double en = 0, ed = 0;
  for (std::size_t i = 0; i < idx.size(); ++i) {
    en += std::norm(cd(u[idx[i]]) - cd(ref[i]));
    ed += std::norm(cd(ref[i]));
  }

  const auto& bs = plan.stats();
  std::printf("init %.3f s: trees %.0f ms, lists %.0f, cones %.0f (far targets %.0f, parent nodes %.0f), upload %.0f "
              "(upward plan %.0f)\n",
              initS, bs.treeMs, bs.listsMs, bs.conesMs, bs.farActivationMs, bs.parentActivationMs, bs.uploadMs, bs.upPlanMs);
  std::printf("device memory %.3f GB (init peak %.3f GB):", bs.deviceBytes / 1e9, bs.initPeakDeviceBytes / 1e9);
  for (const auto& [name, bytes] : plan.deviceMemory()) std::printf(" %s %.3f", name, bytes / 1e9);
  std::printf("\ncones %lld, near pairs %lld, far pairs %lld\n", (long long)st.cones, (long long)st.nearPairs,
              (long long)st.farPairs);
  std::printf("first apply %.1f ms (includes JIT compilation on the generic target)\n", firstMs);
  std::printf("apply: best %.1f ms, mean %.1f ms over %d | near %.1f, leaf %.1f, upward %.1f, far %.1f ms\n", best,
              reps > 0 ? sum / reps : sum, reps, st.nearMs, st.leafMs, st.upwardMs, st.farMs);
  if (st.lookupMisses != 0) std::printf("WARNING: %d far-field lookups hit an inactive cone\n", st.lookupMisses);
  std::printf("relative l2 error: %.3e\n", std::sqrt(en / ed));
}

}  // namespace

int main(int argc, char** argv)
{
  const int N = argc > 1 ? std::atoi(argv[1]) : 100000;
  const double kre = argc > 2 ? std::atof(argv[2]) : 3.14 * 4.0;
  const double kim = argc > 3 ? std::atof(argv[3]) : 0.001;
  const std::string op = argc > 4 ? argv[4] : "SL";
  FacePlanOptions opt;
  opt.leafSize = argc > 5 ? std::atoi(argv[5]) : 64;
  opt.coneC = argc > 6 ? std::atof(argv[6]) : 0.15;
  opt.face.delta = argc > 7 ? std::atof(argv[7]) : 0.0;
  const int reps = argc > 8 ? std::atoi(argv[8]) : 3;
  opt.face.fu = argc > 9 ? std::atof(argv[9]) : 2.25;
  opt.face.fw = argc > 10 ? std::atof(argv[10]) : 1.0;
  const int P = argc > 11 ? std::atoi(argv[11]) : 8;
  if (N < 1 || kre < 0 || kim < 0 || (op != "SL" && op != "DL" && op != "CF") || P < 4 || P > 8 || reps < 0) {
    std::fprintf(stderr, "usage: face_test [N] [kappa.re >= 0] [kappa.im >= 0] [SL|DL|CF] [leafSize] [C] [delta] [reps] "
                         "[fu] [fw] [P = 4..8]\n");
    return 2;
  }
  const cd kappa{kre, kim};
  try {
    sycl::queue q{pickDevice(), sycl::property::queue::in_order{}};
    std::printf("IFGF face_test: N=%d kappa=(%g, %g) operator=%s orders (%d,%d,%d) leafSize=%d C=%g fu=%.2f fw=%.2f "
                "delta=%.2f\ndevice: %s\n",
                N, kre, kim, op.c_str(), P - 2, P, P, opt.leafSize, opt.coneC, opt.face.fu, opt.face.fw, opt.face.delta,
                exec::describe(q.get_device()).c_str());
    const auto p = sphereSurface(static_cast<std::size_t>(N), 12345);
    std::vector<Vec3d> nrm(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) nrm[i] = p[i] * (1.0 / norm(p[i]));
    const bool dl = op == "DL", cf = op == "CF";
    auto byP = [&]<int PP>() {
      using namespace kernels;
      auto byKind = [&]<KappaKind K>() {
        if (cf)
          run<HelmholtzCF<float, K>, HelmholtzCF<double, K>, PP>(q, p, nrm, kappa, opt, reps);
        else if (dl)
          run<HelmholtzDL<float, K>, HelmholtzDL<double, K>, PP>(q, p, nrm, kappa, opt, reps);
        else
          run<HelmholtzSL<float, K>, HelmholtzSL<double, K>, PP>(q, p, nrm, kappa, opt, reps);
      };
      if (kim == 0)
        byKind.template operator()<KappaKind::Oscillatory>();
      else
        byKind.template operator()<KappaKind::General>();
    };
    switch (P) {
      case 4: byP.template operator()<4>(); break;
      case 5: byP.template operator()<5>(); break;
      case 6: byP.template operator()<6>(); break;
      case 7: byP.template operator()<7>(); break;
      default: byP.template operator()<8>(); break;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "face_test: %s\n", e.what());
    return 1;
  }
  return 0;
}
