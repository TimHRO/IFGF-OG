#pragma once

// Helmholtz single layer, double layer and combined field kernels. Host and device.
//
// Convention:  G(r) = e^{i kappa r} / (4 pi r),  kappa = kr + i ki, ki >= 0,
// r = |s - t| for a source s and a target t. Double layer: dG/dn_y with n_y the source normal,
//   DL = e^{i kappa r} / (4 pi r^2) (i kappa - 1/r) ((s - t) . n_y).
//
// The wavenumber kind is a compile-time property:
//   Oscillatory (ki = 0):  no exp at all; complex values
//   PureDecay   (kr = 0):  real kernel e^{-ki r}/(4 pi r); real values (with real weights: W = NC lanes)
//   General:               exp and sincos; complex values
//
// Functions for the IFGF pipeline (evaluation point t, source s, box center xc):
//   direct(t, s)                 the kernel itself (reference, host)
//   direct4piFastRX(r^2, xn)     4 pi times the kernel (near field)
//   expikFast*, rayFactor        the factored kernel G(t, s) / CF(t - xc) (leaf stage)
//   centeredComp(y)              CF(y) = e^{i kappa |y|} / (4 pi |y|), y = t - xc (far field)
//   transferFast(xp, o)          CF(x - xc_child) / CF(x - xc_parent), xp = x - xc_parent,
//                                o = xc_child - xc_parent: converts a child's interpolant to the parent
// Phase differences are computed in the stable form d - dc = (d^2 - dc^2) / (d + dc) with
// d^2 - dc^2 evaluated from small box-local vectors, so FP32 does not cancel.
//
// Self-interaction: points closer than sqrt(selfTol2) contribute 0 (one rule for all kernels).
// Complex products are written out (std::complex operators would call __mulsc3 on devices).

#include <ifgf/core/vec3.hpp>

#include <ifgf/core/compensated.hpp>
#include <ifgf/core/math.hpp>

#include <cmath>
#include <complex>
#include <type_traits>

namespace ifgf::kernels {

enum class KappaKind { Oscillatory, PureDecay, General };

// Kind of a wavenumber: the cheapest kind that represents it exactly.
inline KappaKind classify(std::complex<double> kappa)
{
  if (kappa.imag() == 0.0) return KappaKind::Oscillatory;
  if (kappa.real() == 0.0) return KappaKind::PureDecay;
  return KappaKind::General;
}

template <class Real, KappaKind K>
using KernelValue = std::conditional_t<K == KappaKind::PureDecay, Real, std::complex<Real>>;

namespace detail {

template <class Real>
inline std::complex<Real> cmul(const std::complex<Real>& a, const std::complex<Real>& b)
{
  return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
}
template <class Real>
inline Real cmul(Real a, Real b)
{
  return a * b;
}
template <class Real>
inline std::complex<Real> scale(const std::complex<Real>& a, Real s)
{
  return {a.real() * s, a.imag() * s};
}
template <class Real>
inline Real scale(Real a, Real s)
{
  return a * s;
}

template <class Real>
inline constexpr Real inv4pi = Real(0.0795774715459476678844418816862571882);

}  // namespace detail

// Common part of SL and DL: wavenumber, kind, e^{i kappa x}, self-interaction rule.
template <class Real, KappaKind K>
struct HelmholtzBase {
  static constexpr KappaKind kind = K;
  using Value = KernelValue<Real, K>;
  static constexpr int NC = 1;

  Real kr = 0, ki = 0;  // kappa = kr + i ki (the part the kind does not use is ignored)
  Real selfTol2 = 0;    // squared distance below which two points count as coincident
  Real krLo = 0;        // Re kappa - kr (the rounding of the real part; compensated far-field phase)

  HelmholtzBase() = default;
  HelmholtzBase(std::complex<double> kappa, Real selfTol)
      : kr(static_cast<Real>(kappa.real())), ki(static_cast<Real>(kappa.imag())), selfTol2(selfTol * selfTol),
        krLo(static_cast<Real>(kappa.real() - static_cast<double>(static_cast<Real>(kappa.real()))))
  {
  }

  // e^{i kappa x}
  Value expik(Real x) const
  {
    if constexpr (K == KappaKind::Oscillatory) {
      return {math::cos(kr * x), math::sin(kr * x)};
    } else if constexpr (K == KappaKind::PureDecay) {
      return math::exp(-ki * x);
    } else {
      const Real a = math::exp(-ki * x);
      return {a * math::cos(kr * x), a * math::sin(kr * x)};
    }
  }

  // e^{i kappa r} with approximate hardware math (math::fast; |kr r| <= 100 pi)
  Value expikFast(Real r) const
  {
    constexpr Real log2e = Real(1.44269504088896340736);
    if constexpr (K == KappaKind::Oscillatory) {
      const Real p = kr * r;
      return {math::fast::cos(p), math::fast::sin(p)};
    } else if constexpr (K == KappaKind::PureDecay) {
      return math::fast::exp2(-ki * log2e * r);
    } else {
      const Real a = math::fast::exp2(-ki * log2e * r), p = kr * r;
      return {a * math::fast::cos(p), a * math::fast::sin(p)};
    }
  }

  // e^{i kappa x} with approximate hardware math for any |kr x| (range-reduced sin/cos)
  Value expikFastReduced(Real x) const
  {
    constexpr Real log2e = Real(1.44269504088896340736);
    if constexpr (K == KappaKind::PureDecay) {
      return math::fast::exp2(-ki * log2e * x);
    } else {
      Real sn, cs;
      math::fast::sincosReduced(kr * x, sn, cs);
      if constexpr (K == KappaKind::Oscillatory) {
        return {cs, sn};
      } else {
        const Real a = math::fast::exp2(-ki * log2e * x);
        return {a * cs, a * sn};
      }
    }
  }

  // centred factor from y = yh + yl (hi/lo pair) in compensated FP32 (far field over large distances): |y| and the
  // phase Re(kappa) |y| as hi/lo pairs, reduced mod 2 pi before the rounding, so the phase error is a few eps in
  // absolute terms instead of ~eps kr |y|. Not for coincident points.
  Value centeredComp(const Vec3<Real>& yh, const Vec3<Real>& yl) const
  {
    IFGF_NO_FP_CONTRACT
    constexpr Real log2e = Real(1.44269504088896340736);
    Real rh, rl, rinv;
    comp::normHiLo(yh, yl, rh, rl, rinv);
    const Real amp = detail::inv4pi<Real> * rinv;
    if constexpr (K == KappaKind::PureDecay) {
      return math::fast::exp2(-ki * log2e * rh) * amp;
    } else {
      Real ph, pe, sn, cs;
      comp::twoProd(kr, rh, ph, pe);
      comp::sincosHiLo(ph, pe + (kr * rl + krLo * rh), sn, cs);
      const Real a = K == KappaKind::Oscillatory ? amp : math::fast::exp2(-ki * log2e * rh) * amp;
      return {a * cs, a * sn};
    }
  }

  // transfer with approximate hardware math (upward pass)
  Value transferFast(const Vec3<Real>& xp, const Vec3<Real>& o) const
  {
    const Vec3<Real> xc = xp - o;
    const Real d2 = norm2(xc), dp2 = norm2(xp);
    const Real dinv = invRFast(d2), dpinv = math::fast::rsqrt(dp2);
    const Real d = d2 * dinv, dp = dp2 * dpinv;
    const Real delta = dot(o, o - Real(2) * xp) * math::fast::rcp(d + dp);
    return detail::scale(expikFastReduced(delta), dp * dinv);
  }

  // 1/r with the self-interaction rule (0 below selfTol), without a branch
  Real invRFast(Real r2) const { return r2 < selfTol2 ? Real(0) : math::fast::rsqrt(r2); }

  // i kappa - 1/r
  Value ikMinusInvR(Real invr) const
  {
    if constexpr (K == KappaKind::Oscillatory)
      return {-invr, kr};
    else if constexpr (K == KappaKind::PureDecay)
      return -ki - invr;
    else
      return {-ki - invr, kr};
  }
};

template <class Real, KappaKind K>
struct HelmholtzSL : HelmholtzBase<Real, K> {
  using Base = HelmholtzBase<Real, K>;
  using typename Base::Value;
  struct SrcData {};
  using Base::Base;

  Value direct(const Vec3<Real>& t, const Vec3<Real>& s, const SrcData& = {}) const
  {
    const Real r2 = norm2(s - t);
    if (r2 < this->selfTol2) return Value(0);
    const Real r = math::sqrt(r2);
    return detail::scale(this->expik(r), detail::inv4pi<Real> / r);
  }

  // 4 pi G from r^2 = |s - t|^2 with approximate hardware math (near field; the caller folds 1/(4 pi) into the
  // weights; xn = (s - t).n_y is unused here)
  Value direct4piFastRX(Real r2, Real) const
  {
    const Real rinv = this->invRFast(r2);
    return detail::scale(this->expikFast(r2 * rinv), rinv);
  }
};

template <class Real, KappaKind K>
struct HelmholtzDL : HelmholtzBase<Real, K> {
  using Base = HelmholtzBase<Real, K>;
  using typename Base::Value;
  struct SrcData {
    Vec3<Real> n;  // source normal n_y
  };
  using Base::Base;

  Value direct(const Vec3<Real>& t, const Vec3<Real>& s, const SrcData& src) const
  {
    const Vec3<Real> x = s - t;
    const Real r2 = norm2(x);
    if (r2 < this->selfTol2) return Value(0);
    const Real r = math::sqrt(r2), invr = Real(1) / r;
    const Real geom = detail::inv4pi<Real> * invr * invr * dot(x, src.n);
    return detail::scale(detail::cmul(this->expik(r), this->ikMinusInvR(invr)), geom);
  }

  // 4 pi dG/dn_y from r^2 and xn = (s - t).n_y, both computed by the caller in compensated FP32 (near field)
  Value direct4piFastRX(Real r2, Real xn) const
  {
    const Real rinv = this->invRFast(r2);
    return detail::scale(detail::cmul(this->expikFast(r2 * rinv), this->ikMinusInvR(rinv)), rinv * rinv * xn);
  }

  // leaf ray kernel: factored = e^{i kappa (d - dc)} (dc / d) * rayFactor(1/d, (s - t).n / d)
  Value rayFactor(Real dinv, Real xnd) const { return detail::scale(this->ikMinusInvR(dinv), xnd); }
};

// Combined field (Brakhage-Werner): K = dG/dn_y - i eta G, eta a coupling parameter (default eta = kappa; complex
// when kappa is). With x = s - t, r = |x|:
//   K = e^{i kappa r} / (4 pi r) [ (i kappa - 1/r) (x . n_y) / r - i eta ]
template <class Real, KappaKind K>
struct HelmholtzCF : HelmholtzBase<Real, K> {
  static_assert(K != KappaKind::PureDecay, "HelmholtzCF: complex values (use KappaKind::General for kappa = i ki)");
  using Base = HelmholtzBase<Real, K>;
  using typename Base::Value;
  struct SrcData {
    Vec3<Real> n;  // source normal n_y
  };
  Real er = 0, ei = 0;  // eta = er + i ei

  HelmholtzCF() = default;
  HelmholtzCF(std::complex<double> kappa, Real selfTol) : HelmholtzCF(kappa, selfTol, kappa) {}
  HelmholtzCF(std::complex<double> kappa, Real selfTol, std::complex<double> eta)
      : Base(kappa, selfTol), er(static_cast<Real>(eta.real())), ei(static_cast<Real>(eta.imag()))
  {
  }

  // [(i kappa - 1/r) xn / r - i eta]
  Value bracket(Real rinv, Real xn) const
  {
    const Value a = detail::scale(this->ikMinusInvR(rinv), xn * rinv);
    return {a.real() + ei, a.imag() - er};
  }

  Value direct(const Vec3<Real>& t, const Vec3<Real>& s, const SrcData& src) const
  {
    const Vec3<Real> x = s - t;
    const Real r2 = norm2(x);
    if (r2 < this->selfTol2) return Value(0);
    const Real r = math::sqrt(r2), invr = Real(1) / r;
    return detail::scale(detail::cmul(this->expik(r), bracket(invr, dot(x, src.n))), detail::inv4pi<Real> * invr);
  }

  // 4 pi K from r^2 and xn = (s - t).n_y, both computed by the caller in compensated FP32 (near field)
  Value direct4piFastRX(Real r2, Real xn) const
  {
    const Real rinv = this->invRFast(r2);
    // rinv = 0 (self-interaction) gives 0: the bracket stays finite, the factor rinv vanishes
    return detail::scale(detail::cmul(this->expikFast(r2 * rinv), bracket(rinv, xn)), rinv);
  }

  // leaf ray kernel: factored = e^{i kappa (d - dc)} (dc / d) * rayFactor(1/d, (s - t).n / d)
  Value rayFactor(Real dinv, Real xnd) const
  {
    const Value a = detail::scale(this->ikMinusInvR(dinv), xnd);
    return {a.real() + ei, a.imag() - er};
  }
};

}  // namespace ifgf::kernels
