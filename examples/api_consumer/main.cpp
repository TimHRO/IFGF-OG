// Example: the installed ifgf library from a plain C++17 program (no SYCL). Buffers of interleaved doubles, init,
// mult with complex<float>; errors against a double direct sum at 100 targets. Then the generic interface
// (ifgf::Operator) on vector fields, the hypersingular and the Maxwell kernels.
// usage: api_consumer [N=50000] [kappa.re=10] [kappa.im=0] [components=3]
#include <ifgf/ifgf_library.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

using cd = std::complex<double>;

int main(int argc, char** argv)
{
  const std::size_t N = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 50000;
  const cd kappa{argc > 2 ? std::atof(argv[2]) : 10.0, argc > 3 ? std::atof(argv[3]) : 0.0};
  std::mt19937_64 rng(42);
  std::normal_distribution<double> g;
  std::uniform_real_distribution<double> u(-1, 1);
  std::vector<double> pts(3 * N), nrm(3 * N);
  for (std::size_t i = 0; i < N; ++i) {  // points on the unit sphere, radial normals; sources = targets
    double x = g(rng), y = g(rng), z = g(rng), r = std::sqrt(x * x + y * y + z * z);
    pts[3 * i] = nrm[3 * i] = x / r, pts[3 * i + 1] = nrm[3 * i + 1] = y / r, pts[3 * i + 2] = nrm[3 * i + 2] = z / r;
  }
  std::vector<std::complex<float>> w(N), res(N);
  for (auto& v : w) v = {static_cast<float>(u(rng)), static_cast<float>(u(rng))};

  const double tol2 = 1e-12 * 4;  // self rule: |x - y| < 1e-6 * box side (side ~ 2)
  auto exact = [&](int kind, std::size_t t) {
    cd s = 0;
    for (std::size_t j = 0; j < N; ++j) {
      const double dx = pts[3 * j] - pts[3 * t], dy = pts[3 * j + 1] - pts[3 * t + 1], dz = pts[3 * j + 2] - pts[3 * t + 2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 < tol2) continue;
      const double r = std::sqrt(r2);
      const cd G = std::exp(cd(0, 1) * kappa * r) / (4 * M_PI * r);
      const double xn = dx * nrm[3 * j] + dy * nrm[3 * j + 1] + dz * nrm[3 * j + 2];
      const cd DL = G * (cd(0, 1) * kappa - 1.0 / r) * xn / r;
      const cd K = kind == 0 ? G : kind == 1 ? DL : DL - cd(0, 1) * kappa * G;
      s += K * cd(w[j]);
    }
    return s;
  };
  const char* names[3] = {"SL", "DL", "CF"};
  for (int kind = 0; kind < 3; ++kind) {
    std::unique_ptr<ifgf::HelmholtzOperator3D> op;
    if (kind == 0) op = std::make_unique<ifgf::HelmholtzSL3D>(kappa, 64, 8, 0, 1e-6, 0, 0);
    if (kind == 1) op = std::make_unique<ifgf::HelmholtzDL3D>(kappa, 64, 8, 0, 1e-6, 0, 0);
    if (kind == 2) op = std::make_unique<ifgf::HelmholtzCF3D>(kappa, 64, 8, 0, 1e-6, 0, 0);
    const auto t0 = std::chrono::steady_clock::now();
    if (kind == 0)
      op->init(pts.data(), N, pts.data(), N);
    else
      op->init(pts.data(), N, pts.data(), N, nrm.data(), N);
    const auto t1 = std::chrono::steady_clock::now();
    op->mult(w.data(), w.size(), res.data(), res.size());
    const auto t2 = std::chrono::steady_clock::now();
    double num = 0, den = 0;
    for (std::size_t t = 0; t < N; t += N / 100) {
      const cd e = exact(kind, t);
      num += std::norm(cd(res[t]) - e), den += std::norm(e);
    }
    std::printf("%s: init %.3f s, mult %.1f ms, rel l2 error %.2e\n  %s\n", names[kind],
                std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double, std::milli>(t2 - t1).count(),
                std::sqrt(num / den), op->info().c_str());
  }
  // vector fields: the generic interface with nc components (point-major w[s * nc + c])
  const int nc = argc > 4 ? std::atoi(argv[4]) : 3;
  std::vector<std::complex<float>> wv(N * nc), rv(N * nc);
  for (auto& v : wv) v = {static_cast<float>(u(rng)), static_cast<float>(u(rng))};
  const ifgf::Kernel kinds[3] = {ifgf::Kernel::SingleLayer, ifgf::Kernel::DoubleLayer, ifgf::Kernel::CombinedField};
  ifgf::Options opt;
  opt.leafSize = 64;
  for (int kind = 0; kind < 3; ++kind) {
    ifgf::Operator op(kinds[kind], kappa, nc, opt);
    const auto t0 = std::chrono::steady_clock::now();
    op.init(pts.data(), N, pts.data(), N, nrm.data());
    const auto t1 = std::chrono::steady_clock::now();
    op.mult(wv.data(), wv.size(), rv.data(), rv.size());
    const auto t2 = std::chrono::steady_clock::now();
    double worst = 0;
    for (int c = 0; c < nc; ++c) {  // component c through the scalar reference
      for (std::size_t j = 0; j < N; ++j) w[j] = wv[j * nc + c];
      double num = 0, den = 0;
      for (std::size_t t = 0; t < N; t += N / 100) {
        const cd e = exact(kind, t);
        num += std::norm(cd(rv[t * nc + c]) - e), den += std::norm(e);
      }
      worst = std::max(worst, std::sqrt(num / den));
    }
    std::printf("%s, %d components: init %.3f s, mult %.1f ms, worst component rel l2 error %.2e\n", names[kind], nc,
                std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double, std::milli>(t2 - t1).count(),
                worst);
  }
  // hypersingular (previous interface, 4 values per point), MaxwellSL and MaxwellDL (ngbem kernels.hpp definitions)
  const cd I(0, 1);
  for (int kind = 0; kind < 3; ++kind) {
    const int w4 = kind == 2 ? 3 : 4;
    std::vector<std::complex<float>> wk(N * w4), rk(N * w4);
    for (auto& v : wk) v = {static_cast<float>(u(rng)), static_cast<float>(u(rng))};
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_ptr<ifgf::HelmholtzHS3D> hs;
    std::unique_ptr<ifgf::Operator> mw;
    if (kind == 0) {
      hs = std::make_unique<ifgf::HelmholtzHS3D>(kappa, 64, 8, 0, 1e-6, 0, 0);
      hs->init(pts.data(), N, pts.data(), N, nrm.data(), N, nrm.data(), N);
    } else {
      mw = std::make_unique<ifgf::Operator>(kind == 1 ? ifgf::Kernel::MaxwellSingleLayer : ifgf::Kernel::MaxwellDoubleLayer,
                                            kappa, 1, opt);
      mw->init(pts.data(), N, pts.data(), N);
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (hs) hs->mult(wk.data(), wk.size(), rk.data(), rk.size());
    if (mw) mw->mult(wk.data(), wk.size(), rk.data(), rk.size());
    const auto t2 = std::chrono::steady_clock::now();
    double num = 0, den = 0;
    for (std::size_t t = 0; t < N; t += N / 100) {
      cd acc[4] = {0, 0, 0, 0};
      for (std::size_t j = 0; j < N; ++j) {
        const double d[3] = {pts[3 * t] - pts[3 * j], pts[3 * t + 1] - pts[3 * j + 1], pts[3 * t + 2] - pts[3 * j + 2]};
        const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if (r2 < tol2) continue;
        const double r = std::sqrt(r2);
        const cd G = std::exp(I * kappa * r) / (4 * M_PI * r);
        const cd* v = nullptr;
        cd vv[4];
        for (int c = 0; c < w4; ++c) vv[c] = cd(wk[j * w4 + c]);
        v = vv;
        if (kind == 0) {
          const double nn = nrm[3 * t] * nrm[3 * j] + nrm[3 * t + 1] * nrm[3 * j + 1] + nrm[3 * t + 2] * nrm[3 * j + 2];
          for (int c = 0; c < 3; ++c) acc[c] += G * v[c];
          acc[3] += -kappa * kappa * G * nn * v[3];
        } else if (kind == 1) {
          for (int c = 0; c < 3; ++c) acc[c] += kappa * G * v[c];
          acc[3] += -1.0 / kappa * G * v[3];
        } else {
          const cd gk = G * (I * kappa - 1.0 / r) / r;
          acc[0] += gk * (d[1] * v[2] - d[2] * v[1]);
          acc[1] += gk * (d[2] * v[0] - d[0] * v[2]);
          acc[2] += gk * (d[0] * v[1] - d[1] * v[0]);
        }
      }
      for (int c = 0; c < w4; ++c) num += std::norm(cd(rk[t * w4 + c]) - acc[c]), den += std::norm(acc[c]);
    }
    const char* hn[3] = {"HS (HelmholtzHS3D)", "MaxwellSL", "MaxwellDL"};
    std::printf("%s: init %.3f s, mult %.1f ms, rel l2 error %.2e\n", hn[kind], std::chrono::duration<double>(t1 - t0).count(),
                std::chrono::duration<double, std::milli>(t2 - t1).count(), std::sqrt(num / den));
  }
  return 0;
}
