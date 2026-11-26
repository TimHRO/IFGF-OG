#pragma once

// Internal: type-erased FacePlan behind the library API (ifgf_library.hpp). Not installed.

#include <ifgf/core/vec3.hpp>
#include <ifgf/plan/face_plan.hpp>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <complex>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ifgf::api_detail {

// One scalar operator (one component); the library applies it per component.
struct Engine {
  virtual ~Engine() = default;
  // out[t * outStride + outOffset] = sum_s K w[s * inStride + inOffset], t < numTargets (strided lanes in and out)
  virtual void apply(const std::complex<float>* w, std::size_t inStride, std::size_t inOffset, std::complex<float>* out,
                     std::size_t outStride, std::size_t outOffset) = 0;
  // new source normals (input order) for the next applies (kernels with normals only)
  virtual void setSourceNormals(std::span<const Vec3d> normals) = 0;
  virtual std::string stats() const = 0;
  virtual std::size_t deviceBytes() const = 0;  // device memory held by the plan (DeviceBuffer bytes)
  std::size_t numSources = 0, numTargets = 0;
  double initSeconds = 0;
};

struct Setup {
  sycl::queue* q = nullptr;
  std::span<const Vec3d> src, tgt, nrm;
  std::complex<double> kappa;
  FacePlanOptions opt;
  int P = 8;
};

template <class K, int P>
struct EngineT final : Engine {
  FacePlan<K, P> plan;
  explicit EngineT(const Setup& s) : plan(*s.q, s.src, s.tgt, s.kappa, s.opt, s.nrm)
  {
    numSources = s.src.size(), numTargets = s.tgt.size();
  }
  void apply(const std::complex<float>* w, std::size_t inStride, std::size_t inOffset, std::complex<float>* out,
             std::size_t outStride, std::size_t outOffset) override
  {
    std::vector<std::complex<float>> wc;
    const std::complex<float>* in = w + inOffset;
    if (inStride != 1) {
      wc.resize(numSources);
      for (std::size_t s = 0; s < numSources; ++s) wc[s] = w[s * inStride + inOffset];
      in = wc.data();
    }
    const auto r = plan.apply(std::span<const std::complex<float>>(in, numSources));
    for (std::size_t t = 0; t < numTargets; ++t) out[t * outStride + outOffset] = r[t];
  }
  void setSourceNormals(std::span<const Vec3d> normals) override { plan.setSourceNormals(normals); }
  std::string stats() const override
  {
    const auto& s = plan.stats();
    std::ostringstream o;
    o << "P " << P << ", " << numSources << " sources, " << numTargets << " targets, " << s.cones << " cones | init " << initSeconds
      << " s (trees " << s.treeMs << " ms, lists " << s.listsMs << ", cones " << s.conesMs << ", upload " << s.uploadMs
      << ") | last apply " << s.applyMs << " ms (near " << s.nearMs << ", leaf " << s.leafMs << ", upward " << s.upwardMs
      << ", far " << s.farMs << ") | device memory " << s.deviceBytes / 1e9 << " GB (init peak "
      << s.initPeakDeviceBytes / 1e9 << " GB;";
    for (const auto& [name, bytes] : plan.deviceMemory()) o << " " << name << " " << bytes / 1e9;
    o << ")";
    return o.str();
  }
  std::size_t deviceBytes() const override { return plan.stats().deviceBytes; }
};

// FacePlan<KT<float, kind>, P> for the wavenumber's kind (real kappa: oscillatory; else general) and P = 4 .. 8
template <template <class, kernels::KappaKind> class KT>
std::unique_ptr<Engine> makeEngine(const Setup& s)
{
  using kernels::KappaKind;
  auto byP = [&]<KappaKind K>() -> std::unique_ptr<Engine> {
    switch (s.P) {
      case 4: return std::make_unique<EngineT<KT<float, K>, 4>>(s);
      case 5: return std::make_unique<EngineT<KT<float, K>, 5>>(s);
      case 6: return std::make_unique<EngineT<KT<float, K>, 6>>(s);
      case 7: return std::make_unique<EngineT<KT<float, K>, 7>>(s);
      case 8: return std::make_unique<EngineT<KT<float, K>, 8>>(s);
      default: throw std::invalid_argument("ifgf: order must be 4 ... 8");
    }
  };
  if (s.kappa.imag() == 0.0) return byP.template operator()<KappaKind::Oscillatory>();
  return byP.template operator()<KappaKind::General>();
}

// one translation unit each (engine_sl.cpp, engine_dl.cpp, engine_cf.cpp)
std::unique_ptr<Engine> makeSL(const Setup& s);
std::unique_ptr<Engine> makeDL(const Setup& s);
std::unique_ptr<Engine> makeCF(const Setup& s);

}  // namespace ifgf::api_detail
