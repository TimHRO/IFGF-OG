// Library API (ifgf_library.hpp): parameter checks, device queues, dispatch to the FacePlan engines.
// Components: the scalar engine is applied once per component. Hypersingular and Maxwell operators are sequences
// of scalar SL / DL applies on one plan with maps at the sources and targets.

#include <ifgf/ifgf_library.hpp>

#include "engine.hpp"

#include <ifgf/exec/device.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace ifgf {

namespace api_detail {

struct Impl {
  Kernel kernel;
  std::complex<double> kappa;
  int components;
  Options options;
  std::unique_ptr<Engine> engine;
  std::vector<Vec3d> srcNormals, tgtNormals;  // Hypersingular: kept for the source and target maps
  double lastMultMs = 0;
};

}  // namespace api_detail

namespace {

bool envVerbose()
{
  const char* v = std::getenv("IFGF_VERBOSE");
  return v && *v && *v != '0';
}

// one in-order queue per device and process. device < 0: IFGF_DEVICE, default gpu, CPU if there is no GPU
sycl::queue& libraryQueue(int device, bool verbose)
{
  static std::mutex m;
  // Never destroyed: the map is constructed before the first queue initializes CUDA, so as a static it would be
  // destroyed after the CUDA runtime has unloaded (cudaStreamDestroy/cudaEventDestroy fail with
  // cudaErrorCudartUnloading at exit). The driver releases streams and memory when the process exits.
  static auto& queues = *new std::map<int, std::unique_ptr<sycl::queue>>;
  std::lock_guard<std::mutex> lock(m);
  auto& q = queues[device < 0 ? -1 : device];
  if (!q) {
    sycl::device d;
    if (device >= 0) {
      d = exec::selectDevice(std::to_string(device));
    } else {
      try {
        d = exec::deviceFromEnv("IFGF_DEVICE", "gpu");
      } catch (const std::exception& e) {
        std::cerr << "ifgf: no GPU device (" << e.what() << "), using the CPU\n";
        d = exec::selectDevice("cpu");
      }
    }
    if (verbose) std::cerr << "ifgf: device " << exec::describe(d) << "\n";
    q = std::make_unique<sycl::queue>(d, sycl::property::queue::in_order{});
  }
  return *q;
}

std::vector<Vec3d> points(const double* p, std::size_t n, const char* what)
{
  if (n > 0 && !p) throw std::invalid_argument(std::string("ifgf: null pointer for ") + what);
  std::vector<Vec3d> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = {p[3 * i], p[3 * i + 1], p[3 * i + 2]};
  return v;
}

const char* name(Kernel k)
{
  switch (k) {
    case Kernel::SingleLayer: return "SingleLayer";
    case Kernel::DoubleLayer: return "DoubleLayer";
    case Kernel::CombinedField: return "CombinedField";
    case Kernel::Hypersingular: return "Hypersingular";
    case Kernel::MaxwellSingleLayer: return "MaxwellSingleLayer";
    default: return "MaxwellDoubleLayer";
  }
}

// values per point of the kernels with a fixed width (0: componentwise kernels)
int fixedWidth(Kernel k)
{
  switch (k) {
    case Kernel::Hypersingular:
    case Kernel::MaxwellSingleLayer: return 4;
    case Kernel::MaxwellDoubleLayer: return 3;
    default: return 0;
  }
}

std::vector<Vec3d> axisNormals(std::size_t n, int k)
{
  Vec3d e{0, 0, 0};
  (k == 0 ? e.x : k == 1 ? e.y : e.z) = 1;
  return std::vector<Vec3d>(n, e);
}

void require(bool ok, const std::string& what)
{
  if (!ok) throw std::invalid_argument("ifgf: " + what);
}

void checkOptions(Options& o)
{
  require(o.leafSize >= 1, "leafSize must be >= 1");
  require(o.order >= 4 && o.order <= 8, "order must be 4 ... 8 (got " + std::to_string(o.order) + ")");
  if (o.radialOrder < 0) o.radialOrder = o.order - 2;
  require(o.radialOrder == o.order - 2,
          "radialOrder must be order - 2 (got " + std::to_string(o.radialOrder) + " for order " + std::to_string(o.order) + ")");
  require(o.coneDensity > 0, "coneDensity must be > 0");
  require(o.radialGridFactor > 0 && o.angularGridFactor > 0, "grid factors must be > 0");
  require(o.minRadialElements >= 1 && o.minAngularElements >= 1, "minimum elements must be >= 1");
  require(o.chartExtension >= 0, "chartExtension must be >= 0");
  require(o.coincidenceTolerance >= 0, "coincidenceTolerance must be >= 0");
}

std::string optionsString(const Options& o)
{
  std::ostringstream s;
  s << "leafSize " << o.leafSize << ", order " << o.order << ", radialOrder " << o.radialOrder << ", coneDensity "
    << o.coneDensity << ", grid factors (" << o.radialGridFactor << ", " << o.angularGridFactor << "), min elements ("
    << o.minRadialElements << ", " << o.minAngularElements << "), chartExtension " << o.chartExtension
    << ", coincidenceTolerance " << o.coincidenceTolerance << ", maxk " << o.maxk << ", minSigma " << o.minSigma
    << ", tolerance " << o.tolerance << " (placeholders), device " << o.device;
  return s.str();
}

}  // namespace

// ---- Operator -----------------------------------------------------------------------------------------------------

Operator::Operator(Kernel kernel, std::complex<double> kappa, int components, const Options& options)
    : impl_(std::make_unique<api_detail::Impl>())
{
  require(kernel == Kernel::SingleLayer || kernel == Kernel::DoubleLayer || kernel == Kernel::CombinedField ||
              kernel == Kernel::Hypersingular || kernel == Kernel::MaxwellSingleLayer || kernel == Kernel::MaxwellDoubleLayer,
          "unknown kernel");
  require(components >= 1, "components must be >= 1");
  require(fixedWidth(kernel) == 0 || components == 1,
          std::string(name(kernel)) + " has " + std::to_string(fixedWidth(kernel)) + " values per point; components must be 1");
  require(std::isfinite(kappa.real()) && std::isfinite(kappa.imag()), "kappa must be finite");
  require(kappa.imag() >= 0, "Im(kappa) must be >= 0");
  require(kappa != std::complex<double>(0), "kappa = 0 (Laplace) is not supported");
  impl_->kernel = kernel, impl_->kappa = kappa, impl_->components = components, impl_->options = options;
  checkOptions(impl_->options);
}

Operator::~Operator() = default;
Operator::Operator(Operator&&) noexcept = default;
Operator& Operator::operator=(Operator&&) noexcept = default;

void Operator::init(const double* sources, std::size_t numSources, const double* targets, std::size_t numTargets,
                    const double* sourceNormals, const double* targetNormals)
{
  auto& I = *impl_;
  require(numSources > 0 && numTargets > 0, "no sources or no targets");
  const bool needNormals = I.kernel == Kernel::DoubleLayer || I.kernel == Kernel::CombinedField || I.kernel == Kernel::Hypersingular;
  require(!needNormals || sourceNormals, std::string(name(I.kernel)) + " needs source normals");
  require(I.kernel != Kernel::Hypersingular || targetNormals, "Hypersingular needs target normals");
  const bool verbose = I.options.verbose || envVerbose();
  const auto t0 = std::chrono::steady_clock::now();
  const auto src = points(sources, numSources, "sources"), tgt = points(targets, numTargets, "targets");
  // MaxwellDoubleLayer: a DL plan with constant axis normals (set per apply in mult)
  const auto nrm = needNormals ? points(sourceNormals, numSources, "source normals")
                   : I.kernel == Kernel::MaxwellDoubleLayer ? axisNormals(numSources, 0)
                                                            : std::vector<Vec3d>{};
  const Options& o = I.options;
  api_detail::Setup s;
  s.q = &libraryQueue(o.device, verbose);
  s.src = src, s.tgt = tgt, s.nrm = nrm;
  s.kappa = I.kappa;
  s.P = o.order;
  s.opt.leafSize = static_cast<int>(o.leafSize);
  s.opt.coneC = o.coneDensity;
  s.opt.selfTolRel = o.coincidenceTolerance;
  s.opt.face.fw = o.radialGridFactor;
  s.opt.face.fu = o.angularGridFactor;
  s.opt.face.nwMin = o.minRadialElements;
  s.opt.face.nuMin = o.minAngularElements;
  s.opt.face.delta = o.chartExtension;
  I.engine.reset();
  I.srcNormals.clear(), I.tgtNormals.clear();
  switch (I.kernel) {
    case Kernel::SingleLayer:
    case Kernel::MaxwellSingleLayer: I.engine = api_detail::makeSL(s); break;
    case Kernel::DoubleLayer:
    case Kernel::MaxwellDoubleLayer: I.engine = api_detail::makeDL(s); break;
    case Kernel::Hypersingular:
      I.engine = api_detail::makeSL(s);
      I.srcNormals = nrm;
      I.tgtNormals = points(targetNormals, numTargets, "target normals");
      break;
    default: I.engine = api_detail::makeCF(s); break;
  }
  I.engine->initSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (verbose) std::cerr << "ifgf: init " << info() << "\n";
}

void Operator::mult(const std::complex<float>* in, std::size_t inSize, std::complex<float>* out, std::size_t outSize) const
{
  auto& I = *impl_;
  if (!I.engine) throw std::logic_error("ifgf: mult before init");
  const std::size_t nc = static_cast<std::size_t>(valuesPerPoint());
  require(inSize == nc * I.engine->numSources && outSize == nc * I.engine->numTargets,
          "mult expects " + std::to_string(nc * I.engine->numSources) + " input and " +
              std::to_string(nc * I.engine->numTargets) + " output values (got " + std::to_string(inSize) + ", " +
              std::to_string(outSize) + ")");
  require(in && out, "null pointer in mult");
  const auto t0 = std::chrono::steady_clock::now();
  api_detail::Engine& E = *I.engine;
  const std::size_t ns = E.numSources, nt = E.numTargets;
  const std::complex<double> kappa = I.kappa;
  switch (I.kernel) {
    case Kernel::Hypersingular: {
      // u_c = SL[v_c] (c < 3); u_3 = n_x . SL[-kappa^2 v_3 n_y] (three lanes, target map accumulated in double)
      for (std::size_t c = 0; c < 3; ++c) E.apply(in, 4, c, out, 4, c);
      std::vector<std::complex<float>> w(ns), u(nt);
      std::vector<std::complex<double>> acc(nt);
      const std::complex<double> f = -kappa * kappa;
      for (int k = 0; k < 3; ++k) {
        for (std::size_t s = 0; s < ns; ++s) w[s] = std::complex<float>(f * std::complex<double>(in[4 * s + 3]) * I.srcNormals[s][k]);
        E.apply(w.data(), 1, 0, u.data(), 1, 0);
        for (std::size_t t = 0; t < nt; ++t) acc[t] += I.tgtNormals[t][k] * std::complex<double>(u[t]);
      }
      for (std::size_t t = 0; t < nt; ++t) out[4 * t + 3] = std::complex<float>(acc[t]);
      break;
    }
    case Kernel::MaxwellSingleLayer: {
      // u_c = kappa SL[v_c] (c < 3), u_3 = -(1 / kappa) SL[v_3]
      for (std::size_t c = 0; c < 4; ++c) E.apply(in, 4, c, out, 4, c);
      const std::complex<double> fv = kappa, fs = -1.0 / kappa;
      for (std::size_t t = 0; t < nt; ++t)
        for (std::size_t c = 0; c < 4; ++c) out[4 * t + c] = std::complex<float>((c < 3 ? fv : fs) * std::complex<double>(out[4 * t + c]));
      break;
    }
    case Kernel::MaxwellDoubleLayer: {
      // u = sum grad_x G x v = -sum grad_y G x v; D_k = DL with constant normal e_k (dG/dy_k):
      // u_0 = -D_1[v_2] + D_2[v_1], u_1 = D_0[v_2] - D_2[v_0], u_2 = -D_0[v_1] + D_1[v_0]
      struct Term {
        int in, outC;
        double sign;
      };
      static constexpr Term terms[3][2] = {{{1, 2, -1}, {2, 1, 1}}, {{2, 0, -1}, {0, 2, 1}}, {{1, 0, 1}, {0, 1, -1}}};
      std::vector<std::complex<float>> u(nt);
      std::vector<std::complex<double>> acc(3 * nt);
      for (int k = 0; k < 3; ++k) {
        E.setSourceNormals(axisNormals(ns, k));
        for (const Term& tm : terms[k]) {
          E.apply(in, 3, static_cast<std::size_t>(tm.in), u.data(), 1, 0);
          for (std::size_t t = 0; t < nt; ++t) acc[3 * t + tm.outC] += tm.sign * std::complex<double>(u[t]);
        }
      }
      for (std::size_t i = 0; i < 3 * nt; ++i) out[i] = std::complex<float>(acc[i]);
      break;
    }
    default:
      for (std::size_t c = 0; c < nc; ++c) E.apply(in, nc, c, out, nc, c);
  }
  I.lastMultMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (I.options.verbose || envVerbose()) std::cerr << "ifgf: mult " << info() << "\n";
}

void Operator::mult(const std::complex<double>* in, std::size_t inSize, std::complex<double>* out, std::size_t outSize) const
{
  require(in && out, "null pointer in mult");
  std::vector<std::complex<float>> inf(inSize), outf(outSize);
  for (std::size_t i = 0; i < inSize; ++i) inf[i] = std::complex<float>(in[i]);
  mult(inf.data(), inSize, outf.data(), outSize);
  for (std::size_t i = 0; i < outSize; ++i) out[i] = std::complex<double>(outf[i]);
}

Kernel Operator::kernel() const { return impl_->kernel; }
std::complex<double> Operator::kappa() const { return impl_->kappa; }
int Operator::components() const { return impl_->components; }
int Operator::valuesPerPoint() const
{
  const int w = fixedWidth(impl_->kernel);
  return w > 0 ? w : impl_->components;
}
const Options& Operator::options() const { return impl_->options; }
bool Operator::initialized() const { return impl_->engine != nullptr; }
std::size_t Operator::numSources() const { return impl_->engine ? impl_->engine->numSources : 0; }
std::size_t Operator::numTargets() const { return impl_->engine ? impl_->engine->numTargets : 0; }
std::size_t Operator::deviceBytes() const { return impl_->engine ? impl_->engine->deviceBytes() : 0; }

std::string Operator::info() const
{
  const auto& I = *impl_;
  std::ostringstream s;
  s << name(I.kernel) << " kappa " << I.kappa << ", " << I.components << " component(s) | " << optionsString(I.options);
  if (I.engine) {
    s << " | " << I.engine->stats();
    if (I.lastMultMs > 0) s << " | last mult " << I.lastMultMs << " ms (all components)";
  } else {
    s << " | not initialised";
  }
  return s.str();
}

// ---- Interface of the previous library ----------------------------------------------------------------------------

namespace {

Kernel kernelOf(HelmholtzKernel k)
{
  switch (k) {
    case HelmholtzKernel::SingleLayer: return Kernel::SingleLayer;
    case HelmholtzKernel::DoubleLayer: return Kernel::DoubleLayer;
    case HelmholtzKernel::CombinedField: return Kernel::CombinedField;
    default: return Kernel::Hypersingular;
  }
}

}  // namespace

HelmholtzOperator3D::HelmholtzOperator3D(HelmholtzKernel kernel, std::complex<double> kappa, std::size_t leafSize,
                                         std::size_t order, int /*nElements*/, double tolerance, double maxk, double minSigma)
    : kernel_(kernel)
{
  Options o;
  o.leafSize = leafSize;
  o.order = static_cast<int>(order);
  o.tolerance = tolerance, o.maxk = maxk, o.minSigma = minSigma;
  op_ = std::make_unique<Operator>(kernelOf(kernel), kappa, 1, o);
}

HelmholtzOperator3D::~HelmholtzOperator3D() = default;
HelmholtzOperator3D::HelmholtzOperator3D(HelmholtzOperator3D&&) noexcept = default;
HelmholtzOperator3D& HelmholtzOperator3D::operator=(HelmholtzOperator3D&&) noexcept = default;

void HelmholtzOperator3D::init(const double* sources, std::size_t numSources, const double* targets, std::size_t numTargets,
                               const double* sourceNormals, std::size_t numSourceNormals, const double* targetNormals,
                               std::size_t numTargetNormals)
{
  if (kernel_ != HelmholtzKernel::SingleLayer)
    require(numSourceNormals == numSources, "one normal per source required (got " + std::to_string(numSourceNormals) +
                                                " for " + std::to_string(numSources) + " sources)");
  if (kernel_ == HelmholtzKernel::Hypersingular)
    require(numTargetNormals == numTargets, "one normal per target required (got " + std::to_string(numTargetNormals) +
                                                " for " + std::to_string(numTargets) + " targets)");
  op_->init(sources, numSources, targets, numTargets, sourceNormals, kernel_ == HelmholtzKernel::Hypersingular ? targetNormals : nullptr);
}

void HelmholtzOperator3D::mult(const std::complex<float>* weights, std::size_t numWeights, std::complex<float>* result,
                               std::size_t numResults) const
{
  op_->mult(weights, numWeights, result, numResults);
}

std::size_t HelmholtzOperator3D::numSources() const { return op_->numSources(); }
std::size_t HelmholtzOperator3D::numTargets() const { return op_->numTargets(); }
std::string HelmholtzOperator3D::info() const { return op_->info(); }

}  // namespace ifgf
