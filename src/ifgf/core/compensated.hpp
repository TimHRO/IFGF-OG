#pragma once

// Error-free transformations for FP32 device code (near-field geometry, far-field phase).
//   twoSum(a, b) = s + e exactly (Knuth, no branch); twoProd(a, b) = p + e exactly (FMA).
//   dot2pair(x, n) for x = xh + xl, n = nh + nl: x.n as an unevaluated pair h + l, accurate to ~eps^2 sum|x_i n_i|
//   (Ogita-Rump-Oishi Dot2 on the hi parts without the final rounding, the lo corrections in plain FP32).
// These need IEEE arithmetic without contraction: IFGF_NO_FP_CONTRACT in every function (host), and products
// formed as FMA nodes (device: the CUDA backend contracts despite the pragma, see twoProd).

#include <ifgf/core/math.hpp>
#include <ifgf/core/vec3.hpp>

namespace ifgf::comp {

template <class Real>
IFGF_INLINE void twoSum(Real a, Real b, Real& s, Real& e)
{
  IFGF_NO_FP_CONTRACT
  s = a + b;
  const Real bb = s - a;
  e = (a - (s - bb)) + (b - bb);
}

// p = fma(a, b, 0), not a * b: the CUDA backend fuses a multiply into a following add regardless of
// "fp contract(off)"; s = p1 + p2 would become fma(a, b, p2) and the error terms would no longer be exact.
// An FMA node is not fused again; fma(a, b, +0) = fl(a b) (up to the sign of a zero product).
template <class Real>
IFGF_INLINE void twoProd(Real a, Real b, Real& p, Real& e)
{
  IFGF_NO_FP_CONTRACT
  p = math::fma(a, b, Real(0));
  e = math::fma(a, b, -p);
}

// (xh + xl) . (nh + nl) as an unevaluated pair h + l (|l| <= ~eps |h|): for differences of two such dot products
template <class Real>
IFGF_INLINE void dot2pair(const Vec3<Real>& xh, const Vec3<Real>& xl, const Vec3<Real>& nh, const Vec3<Real>& nl, Real& h, Real& l)
{
  IFGF_NO_FP_CONTRACT
  Real p1, e1, p2, e2, p3, e3, s, q, q2;
  twoProd(xh.x, nh.x, p1, e1);
  twoProd(xh.y, nh.y, p2, e2);
  twoProd(xh.z, nh.z, p3, e3);
  twoSum(p1, p2, s, q);
  twoSum(s, p3, h, q2);
  l = (e1 + e2 + e3 + q + q2) + (xl.x * nh.x + xl.y * nh.y + xl.z * nh.z) + (xh.x * nl.x + xh.y * nl.y + xh.z * nl.z);
}

// |yh + yl| as rh + rl (|rl| <= ~eps rh, accurate to a few eps^2 rh) and rinv ~ 1 / rh (approximate rsqrt, relative
// error ~2^-23; only for amplitudes). Far-field distances: the phase kappa |y| needs |y| beyond FP32.
template <class Real>
IFGF_INLINE void normHiLo(const Vec3<Real>& yh, const Vec3<Real>& yl, Real& rh, Real& rl, Real& rinv)
{
  IFGF_NO_FP_CONTRACT
  Real h, l, p, e;
  dot2pair(yh, yl, yh, yl, h, l);
  rinv = math::fast::rsqrt(h);
  rh = h * rinv;
  twoProd(rh, rh, p, e);                          // h - p is exact (Sterbenz: p within a few ulp of h)
  rl = (((h - p) - e) + l) * (Real(0.5) * rinv);  // one Newton step on the pair
}

// sin and cos of the phase ph + pl (|pl| <= ~eps |ph|, |ph| < 4e5): Cody-Waite reduction with 2 pi in three parts
// (p1 with 8 significant bits, so k p1 is exact for k < 2^16 and ph - k p1 is exact), the lo part added after the
// reduction, then the MUFU pair on [-pi, pi] (absolute error ~2^-20.5). Phase error: a few eps |r| + |k| 2e-11.
IFGF_INLINE void sincosHiLo(float ph, float pl, float& s, float& c)
{
  IFGF_NO_FP_CONTRACT
  constexpr float inv2pi = 0.159154943091895335769f;
  constexpr float p1 = 6.28125f, p2 = 1.9353071693331003e-3f, p3 = 1.0253376273028358e-11f;  // p1 + p2 + p3 = 2 pi
  const float k = math::rint(ph * inv2pi);
  const float r = ((ph - k * p1) - k * p2) + (pl - k * p3);
  s = math::fast::sin(r);
  c = math::fast::cos(r);
}
inline void sincosHiLo(double ph, double pl, double& s, double& c)
{
  s = math::sin(ph + pl);
  c = math::cos(ph + pl);
}

// |xh + xl|^2 (no cancellation: plain FP32 with the first-order lo term)
template <class Real>
IFGF_INLINE Real norm2c(const Vec3<Real>& xh, const Vec3<Real>& xl)
{
  return norm2(xh) + Real(2) * dot(xh, xl);
}

}  // namespace ifgf::comp
