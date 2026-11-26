#pragma once

// Math functions for code shared by host and device (cone geometry, kernels).
//
// The ahead-of-time CUDA target does not map std::sin/cos/exp in device code to the CUDA math
// library; sycl:: math works on every AdaptiveCpp target, host included. So code that AdaptiveCpp
// compiles (__ACPP__) uses sycl::, plain host code uses std::. Host and device results may differ by
// a few ulps; cone activation covers that with a border margin.

#include <cmath>

#if defined(__ACPP__)
#include <sycl/sycl.hpp>
#define IFGF_MATH_NS sycl
#else
#define IFGF_MATH_NS std
#endif

// Disables FMA contraction in a scope (host): geometry that host and device must compute identically.
#if defined(__clang__)
#define IFGF_NO_FP_CONTRACT _Pragma("clang fp contract(off)")
#else
#define IFGF_NO_FP_CONTRACT
#endif

// Forces inlining of device helpers that take pointers into local memory: if such a helper is
// not inlined, the compiler loses the address space and emits generic loads.
#if defined(__clang__) || defined(__GNUC__)
#define IFGF_INLINE __attribute__((always_inline)) inline
#else
#define IFGF_INLINE inline
#endif

namespace ifgf::math {

template <class Real>
inline Real sqrt(Real x)
{
  return IFGF_MATH_NS::sqrt(x);
}
template <class Real>
inline Real sin(Real x)
{
  return IFGF_MATH_NS::sin(x);
}
template <class Real>
inline Real cos(Real x)
{
  return IFGF_MATH_NS::cos(x);
}
template <class Real>
inline Real exp(Real x)
{
  return IFGF_MATH_NS::exp(x);
}
template <class Real>
inline Real floor(Real x)
{
  return IFGF_MATH_NS::floor(x);
}
template <class Real>
inline Real rint(Real x)
{
  return IFGF_MATH_NS::rint(x);
}
template <class Real>
inline Real fmin(Real a, Real b)
{
  return IFGF_MATH_NS::fmin(a, b);
}
template <class Real>
inline Real fmax(Real a, Real b)
{
  return IFGF_MATH_NS::fmax(a, b);
}
template <class Real>
inline Real fma(Real a, Real b, Real c)
{
  return IFGF_MATH_NS::fma(a, b, c);
}

// Approximate hardware math for throughput-bound kernels. On the CUDA device pass one
// MUFU instruction each (flush-to-zero, no IEEE fixups):
//   rsqrt  rel. error ~2^-23; sin/cos abs. error ~2^-20.5 for |x| <= 100 pi (PTX sin.approx); exp2 ~2 ulp.
// Callers keep their arguments away from denormals and, for sin/cos, within |x| <= 100 pi.
// Elsewhere (host, other targets) sycl::native, or std:: on the plain host library.
namespace fast {

inline float rsqrt(float x)
{
#if defined(__ACPP__) && ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA
  return __nvvm_rsqrt_approx_ftz_f(x);
#elif defined(__ACPP__)
  return sycl::native::rsqrt(x);
#else
  return 1.0f / std::sqrt(x);
#endif
}
inline float sin(float x)
{
#if defined(__ACPP__) && ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA
  return __nvvm_sin_approx_ftz_f(x);
#elif defined(__ACPP__)
  return sycl::native::sin(x);
#else
  return std::sin(x);
#endif
}
inline float cos(float x)
{
#if defined(__ACPP__) && ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA
  return __nvvm_cos_approx_ftz_f(x);
#elif defined(__ACPP__)
  return sycl::native::cos(x);
#else
  return std::cos(x);
#endif
}
inline float exp2(float x)
{
#if defined(__ACPP__) && ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA
  return __nvvm_ex2_approx_ftz_f(x);
#elif defined(__ACPP__)
  return sycl::native::exp2(x);
#else
  return std::exp2(x);
#endif
}
inline float rcp(float x)
{
#if defined(__ACPP__) && ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA
  return __nvvm_rcp_approx_ftz_f(x);
#elif defined(__ACPP__)
  return sycl::native::recip(x);
#else
  return 1.0f / x;
#endif
}
// sin and cos for |x| < 4e5 (k < 2^16, so k * twoPiHi is exact): Cody-Waite reduction to [-pi, pi],
// then the MUFU pair. The error is that of the MUFU on [-pi, pi] plus the rounding of x itself.
inline void sincosReduced(float x, float& s, float& c)
{
  constexpr float inv2pi = 0.159154943091895335769f;
  constexpr float twoPiHi = 6.28125f;                    // 8 significant bits
  constexpr float twoPiLo = 1.93530717958647692528e-3f;  // 2 pi - twoPiHi
  const float k = IFGF_MATH_NS::rint(x * inv2pi);
  const float r = (x - k * twoPiHi) - k * twoPiLo;
  s = sin(r);
  c = cos(r);
}
// double: no fast path (host reference code only)
inline double rcp(double x) { return 1.0 / x; }
inline void sincosReduced(double x, double& s, double& c)
{
  s = IFGF_MATH_NS::sin(x);
  c = IFGF_MATH_NS::cos(x);
}
inline double rsqrt(double x) { return 1.0 / IFGF_MATH_NS::sqrt(x); }
inline double sin(double x) { return IFGF_MATH_NS::sin(x); }
inline double cos(double x) { return IFGF_MATH_NS::cos(x); }
inline double exp2(double x) { return IFGF_MATH_NS::exp2(x); }

}  // namespace fast

}  // namespace ifgf::math

#undef IFGF_MATH_NS
