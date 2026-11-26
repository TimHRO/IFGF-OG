#pragma once

// ifgf library API: boundary operators of the Helmholtz kernel in 3D, evaluated with the face-chart IFGF (FP32 on one
// device). Plain C++17 header: no SYCL and no internal headers (the implementation lives in libifgf_library, built
// with AdaptiveCpp). Install with CMake and use as
//   find_package(ifgf CONFIG REQUIRED)
//   target_link_libraries(<target> PRIVATE ifgf::ifgf_library)
//
// Kernels (x = target, y = source, n_y = source normal, kappa = kr + i ki with ki >= 0, G = e^{i kappa r} / (4 pi r)):
//   SingleLayer    u_c(x) = sum_y G(x, y) w_c(y)
//   DoubleLayer    u_c(x) = sum_y dG/dn_y(x, y) w_c(y)
//   CombinedField  u_c(x) = sum_y (dG/dn_y - i kappa G)(x, y) w_c(y)
// Each of these acts componentwise on NC = `components` values per point (vector fields: NC = 3).
// Kernels with a fixed number of values per point (as ngbem's HelmholtzHSKernel<3>, MaxwellSLKernel<3>,
// MaxwellDLKernel<3>; `components` must be 1, valuesPerPoint() gives the width; n_x = target normal):
//   Hypersingular       4 values: u_c(x) = sum_y G v_c (c = 0, 1, 2),  u_3(x) = n_x . sum_y G (-kappa^2 v_3 n_y)
//                       (source and target normals required)
//   MaxwellSingleLayer  4 values: u_c(x) = kappa sum_y G v_c (c = 0, 1, 2),  u_3(x) = -(1 / kappa) sum_y G v_3
//   MaxwellDoubleLayer  3 values: u(x) = sum_y grad_x G(x - y) x v(y)   (no normals)
// They are evaluated as sequential scalar applies on one plan (hypersingular 6, MaxwellSL 4, MaxwellDL 6).
// Pairs closer than Options::coincidenceTolerance * bounding box side contribute 0.
//
// Usage:
//   ifgf::Options opt;                       // every ifgf parameter, see below
//   ifgf::Operator op(ifgf::Kernel::DoubleLayer, kappa, 3, opt);
//   op.init(src, numSrc, tgt, numTgt, srcNormals);   // xyz interleaved doubles; copied
//   op.mult(w, 3 * numSrc, u, 3 * numTgt);           // point-major: w[s * NC + c], u[t * NC + c]
//   (NC = valuesPerPoint(): components, or the width of Hypersingular / MaxwellSingleLayer / MaxwellDoubleLayer)
//
// Environment: IFGF_DEVICE = gpu (default) | cpu | <device index> when Options::device < 0; IFGF_VERBOSE=1 prints
// setup and apply timings (as Options::verbose).

#include <complex>
#include <cstddef>
#include <memory>
#include <string>

#if defined(_WIN32)
#define IFGF_API
#else
#define IFGF_API __attribute__((visibility("default")))
#endif

namespace ifgf {

namespace api_detail {
struct Impl;
}

enum class Kernel { SingleLayer, DoubleLayer, CombinedField, Hypersingular, MaxwellSingleLayer, MaxwellDoubleLayer };

// Parameters of the method. Negative values of radialOrder, maxk, minSigma and tolerance mean "not set".
struct Options {
  std::size_t leafSize = 800;          // maximum number of points per leaf box
  int order = 8;                       // angular interpolation order P (4 ... 8)
  int radialOrder = -1;                // radial interpolation order; -1: order - 2 (the only one supported so far)
  double coneDensity = 0.15;           // cone elements per box side and unit |kappa|
  double radialGridFactor = 1.0;       // radial elements of a face cone grid, factor on the density rule
  double angularGridFactor = 2.25;     // angular elements of a face cone grid, factor on the density rule
  int minRadialElements = 2;           // minimum face cone grid, radial
  int minAngularElements = 4;          // minimum face cone grid, per angular axis
  double chartExtension = 0.0;         // overlap of the face charts in cells (0: fallback lines at the edges)
  double coincidenceTolerance = 1e-6;  // pairs closer than this * bounding box side contribute 0
  // Placeholders for convolution quadrature (reuse of one init for many wavenumbers): the wavenumber range of an init
  // (maxk: largest oscillation |Re kappa|, minSigma: smallest decay Im kappa) and the decay cutoff of far interactions
  // (dropped where e^{-dist Im kappa} < tolerance). Accepted and reported; no effect yet.
  double maxk = -1, minSigma = -1, tolerance = -1;
  int device = -1;                     // device index; -1: IFGF_DEVICE, else the GPU (CPU if there is none)
  bool verbose = false;                // timings to stderr (also IFGF_VERBOSE=1)
};

class IFGF_API Operator {
public:
  // components: values per point (1 scalar, 3 vector field, ...) for SingleLayer, DoubleLayer, CombinedField; must be 1
  // for the kernels with a fixed width (Hypersingular 4, MaxwellSingleLayer 4, MaxwellDoubleLayer 3)
  Operator(Kernel kernel, std::complex<double> kappa, int components = 1, const Options& options = Options{});
  ~Operator();
  Operator(Operator&&) noexcept;
  Operator& operator=(Operator&&) noexcept;
  Operator(const Operator&) = delete;
  Operator& operator=(const Operator&) = delete;

  // Builds the operator (trees, cones, plans). Points and normals as interleaved xyz doubles, copied. Source normals
  // (one per source) are required by DoubleLayer, CombinedField and Hypersingular, target normals (one per target) by
  // Hypersingular; the other kernels ignore them.
  void init(const double* sources, std::size_t numSources, const double* targets, std::size_t numTargets,
            const double* sourceNormals = nullptr, const double* targetNormals = nullptr);

  // out = K in. Point-major: in[s * NC + c] with inSize == NC * numSources, out[t * NC + c] with
  // outSize == NC * numTargets, NC = valuesPerPoint(); out is overwritten. Computed in FP32; the double overload
  // converts.
  void mult(const std::complex<float>* in, std::size_t inSize, std::complex<float>* out, std::size_t outSize) const;
  void mult(const std::complex<double>* in, std::size_t inSize, std::complex<double>* out, std::size_t outSize) const;

  Kernel kernel() const;
  std::complex<double> kappa() const;
  int components() const;
  int valuesPerPoint() const;  // NC of mult: components, or the fixed width of the kernel
  const Options& options() const;  // as given, with derived values filled in (radialOrder)
  bool initialized() const;
  std::size_t numSources() const;
  std::size_t numTargets() const;
  // device memory held by this operator in bytes (0 before init); the per-buffer breakdown is in info()
  std::size_t deviceBytes() const;
  // parameters, sizes, device memory and timings of the last init / mult (human readable)
  std::string info() const;

private:
  std::unique_ptr<api_detail::Impl> impl_;
};

// ---- Interface of the previous library (kept for existing callers) ------------------------------------------------
// HelmholtzSL3D / DL3D / CF3D = Operator with one component; tolerance, maxk and minSigma go to Options, nElements is
// ignored. HelmholtzHS3D is constructible, its init throws std::runtime_error (not implemented yet).

enum class HelmholtzKernel { SingleLayer, DoubleLayer, CombinedField, Hypersingular };

class IFGF_API HelmholtzOperator3D {
public:
  HelmholtzOperator3D(HelmholtzKernel kernel, std::complex<double> kappa, std::size_t leafSize, std::size_t order,
                      int nElements = 0, double tolerance = -1, double maxk = -1, double minSigma = -1);
  ~HelmholtzOperator3D();
  HelmholtzOperator3D(HelmholtzOperator3D&&) noexcept;
  HelmholtzOperator3D& operator=(HelmholtzOperator3D&&) noexcept;
  HelmholtzOperator3D(const HelmholtzOperator3D&) = delete;
  HelmholtzOperator3D& operator=(const HelmholtzOperator3D&) = delete;

  // Source normals are required for the double layer and the combined field (numSourceNormals == numSources);
  // target normals only for the hypersingular operator.
  void init(const double* sources, std::size_t numSources, const double* targets, std::size_t numTargets,
            const double* sourceNormals = nullptr, std::size_t numSourceNormals = 0, const double* targetNormals = nullptr,
            std::size_t numTargetNormals = 0);

  // result[t] = sum_s K(target t, source s) weights[s]; numWeights == numSources, numResults == numTargets.
  void mult(const std::complex<float>* weights, std::size_t numWeights, std::complex<float>* result,
            std::size_t numResults) const;

  std::size_t numSources() const;
  std::size_t numTargets() const;
  std::string info() const;

private:
  HelmholtzKernel kernel_;
  std::unique_ptr<Operator> op_;
};

class IFGF_API HelmholtzSL3D : public HelmholtzOperator3D {
public:
  HelmholtzSL3D(std::complex<double> kappa, std::size_t leafSize, std::size_t order, int nElements = 0, double tolerance = -1,
                double maxk = -1, double minSigma = -1)
      : HelmholtzOperator3D(HelmholtzKernel::SingleLayer, kappa, leafSize, order, nElements, tolerance, maxk, minSigma)
  {
  }
};

class IFGF_API HelmholtzDL3D : public HelmholtzOperator3D {
public:
  HelmholtzDL3D(std::complex<double> kappa, std::size_t leafSize, std::size_t order, int nElements = 0, double tolerance = -1,
                double maxk = -1, double minSigma = -1)
      : HelmholtzOperator3D(HelmholtzKernel::DoubleLayer, kappa, leafSize, order, nElements, tolerance, maxk, minSigma)
  {
  }
};

class IFGF_API HelmholtzCF3D : public HelmholtzOperator3D {
public:
  HelmholtzCF3D(std::complex<double> kappa, std::size_t leafSize, std::size_t order, int nElements = 0, double tolerance = -1,
                double maxk = -1, double minSigma = -1)
      : HelmholtzOperator3D(HelmholtzKernel::CombinedField, kappa, leafSize, order, nElements, tolerance, maxk, minSigma)
  {
  }
};

class IFGF_API HelmholtzHS3D : public HelmholtzOperator3D {
public:
  HelmholtzHS3D(std::complex<double> kappa, std::size_t leafSize, std::size_t order, int nElements = 0, double tolerance = -1,
                double maxk = -1, double minSigma = -1)
      : HelmholtzOperator3D(HelmholtzKernel::Hypersingular, kappa, leafSize, order, nElements, tolerance, maxk, minSigma)
  {
  }
};

}  // namespace ifgf
