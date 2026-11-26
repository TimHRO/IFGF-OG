#pragma once

// Evaluation of a cone interpolant at R points per thread, coefficients in local memory (far field). Explicit
// Chebyshev values and sum factorisation: a0 inner (fully unrolled, T_a0 in registers), then a1 and a2 with the
// T recurrences on the fly. The radial order is a compile-time constant; the lanes are read with vector loads.

#include <ifgf/core/math.hpp>
#include <ifgf/interp/orders.hpp>

#include <sycl/sycl.hpp>

#include <type_traits>

namespace ifgf::interp {

namespace detail {

// Loads W consecutive lanes with the widest aligned vector loads (LDS.128/.64 on NVIDIA).
template <int W, class Real>
IFGF_INLINE void loadLanes(const Real* p, Real c[W])
{
  if constexpr (W % 4 == 0) {
#pragma unroll
    for (int k = 0; k < W; k += 4) {
      const sycl::vec<Real, 4> v = *reinterpret_cast<const sycl::vec<Real, 4>*>(p + k);
      c[k] = v[0], c[k + 1] = v[1], c[k + 2] = v[2], c[k + 3] = v[3];
    }
  } else if constexpr (W % 2 == 0) {
#pragma unroll
    for (int k = 0; k < W; k += 2) {
      const sycl::vec<Real, 2> v = *reinterpret_cast<const sycl::vec<Real, 2>*>(p + k);
      c[k] = v[0], c[k + 1] = v[1];
    }
  } else {
#pragma unroll
    for (int k = 0; k < W; ++k) c[k] = p[k];
  }
}

// Loads the lanes of two consecutive a0 coefficients (2W values) with the widest vector loads the
// alignment allows: LDS.128 for W = 2 (one load instead of two LDS.64), LDS.64 pairs for W = 1.
template <int W, class Real>
IFGF_INLINE void loadLanes2(const Real* p, Real c[2 * W])
{
  if constexpr ((2 * W) % 4 == 0) {
#pragma unroll
    for (int k = 0; k < 2 * W; k += 4) {
      const sycl::vec<Real, 4> v = *reinterpret_cast<const sycl::vec<Real, 4>*>(p + k);
      c[k] = v[0], c[k + 1] = v[1], c[k + 2] = v[2], c[k + 3] = v[3];
    }
  } else {
#pragma unroll
    for (int k = 0; k < 2 * W; k += 2) {
      const sycl::vec<Real, 2> v = *reinterpret_cast<const sycl::vec<Real, 2>*>(p + k);
      c[k] = v[0], c[k + 1] = v[1];
    }
  }
}

// Pairwise a0 loads are legal when every pair starts at the vector alignment: the cone base is 16 B
// aligned (staged vec4 buffers; coef slots of len = W * nodes floats) and rows are W * N0 floats.
template <int W, int N0>
inline constexpr bool evalPairLoads = N0 > 0 && N0 % 2 == 0 && ((2 * W) % 4 == 0 ? (W * N0) % 4 == 0 : true);

// Core: evaluates the cone in local memory at R points per thread, given in registers.
// lc: the cone's coefficients in local memory, [a][W]; xs[j]: local coordinates of point j;
// res[j]: its W lanes. N0 > 0: radial order known at compile time (no guards in the inner loop);
// N0 = 0: runtime n0 <= maxOrder with guards (fallback for unusual orders).
// N12 > 0: n1 = n2 = N12 at compile time. The a1 loop is unrolled by 4, so the T recurrence rotates by
// renaming and the loop overhead is shared by 4 a1 steps.
template <int W, int R, int N0, class Real, int N12 = 0>
IFGF_INLINE void evalAtPoints(const Real* lc, const Real (&xs)[R][3], Real (&res)[R][W], const Orders& o)
{
  constexpr int NI = N0 > 0 ? N0 : maxOrder;  // inner loop length
  constexpr bool PAIR = evalPairLoads<W, N0>;
  const int n0 = N0 > 0 ? N0 : o.n[0];
  const int n1 = N12 > 0 ? N12 : o.n[1], n2 = N12 > 0 ? N12 : o.n[2];

  // per point: T_a0 values, and the coordinates for the a1/a2 recurrences
  Real t0[R][NI], x1[R], x2[R];
#pragma unroll
  for (int j = 0; j < R; ++j) {
    const Real x0 = xs[j][0], tx = Real(2) * xs[j][0];
    t0[j][0] = Real(1);
    if (NI > 1) t0[j][1] = x0;
#pragma unroll
    for (int k = 2; k < NI; ++k) t0[j][k] = tx * t0[j][k - 1] - t0[j][k - 2];
    x1[j] = xs[j][1];
    x2[j] = xs[j][2];
  }

  Real acc2[R][W];
#pragma unroll
  for (int j = 0; j < R; ++j)
#pragma unroll
    for (int k = 0; k < W; ++k) acc2[j][k] = Real(0);
  // T_a recurrences on the fly: T_{-1} := x so that T_1 = 2x*T_0 - T_{-1} = x
  Real t2c[R], t2p[R];
#pragma unroll
  for (int j = 0; j < R; ++j) t2c[j] = Real(1), t2p[j] = x2[j];

#pragma unroll 1
  for (int a2 = 0; a2 < n2; ++a2) {
    Real acc1[R][W];
#pragma unroll
    for (int j = 0; j < R; ++j)
#pragma unroll
      for (int k = 0; k < W; ++k) acc1[j][k] = Real(0);
    Real t1c[R], t1p[R];
#pragma unroll
    for (int j = 0; j < R; ++j) t1c[j] = Real(1), t1p[j] = x1[j];

#pragma unroll 4
    for (int a1 = 0; a1 < n1; ++a1) {
      const Real* c0 = lc + W * n0 * (a1 + n1 * a2);
      Real acc0[R][W];
      if constexpr (PAIR) {
#pragma unroll
        for (int a0 = 0; a0 < N0; a0 += 2) {
          Real c[2 * W];
          loadLanes2<W>(c0 + W * a0, c);
#pragma unroll
          for (int j = 0; j < R; ++j)
#pragma unroll
            for (int k = 0; k < W; ++k) {
              acc0[j][k] = a0 == 0 ? c[k] : acc0[j][k] + t0[j][a0] * c[k];  // T_0 = 1
              acc0[j][k] = acc0[j][k] + t0[j][a0 + 1] * c[W + k];
            }
        }
      } else {
#pragma unroll
        for (int a0 = 0; a0 < NI; ++a0) {
          if (N0 > 0 || a0 < n0) {
            Real c[W];
            loadLanes<W>(c0 + W * a0, c);
#pragma unroll
            for (int j = 0; j < R; ++j)
#pragma unroll
              for (int k = 0; k < W; ++k) acc0[j][k] = a0 == 0 ? c[k] : acc0[j][k] + t0[j][a0] * c[k];  // T_0 = 1
          }
        }
      }
#pragma unroll
      for (int j = 0; j < R; ++j) {
#pragma unroll
        for (int k = 0; k < W; ++k) acc1[j][k] += t1c[j] * acc0[j][k];
        const Real t1n = Real(2) * x1[j] * t1c[j] - t1p[j];
        t1p[j] = t1c[j];
        t1c[j] = t1n;
      }
    }
#pragma unroll
    for (int j = 0; j < R; ++j) {
#pragma unroll
      for (int k = 0; k < W; ++k) acc2[j][k] += t2c[j] * acc1[j][k];
      const Real t2n = Real(2) * x2[j] * t2c[j] - t2p[j];
      t2p[j] = t2c[j];
      t2c[j] = t2n;
    }
  }
#pragma unroll
  for (int j = 0; j < R; ++j)
#pragma unroll
    for (int k = 0; k < W; ++k) res[j][k] = acc2[j][k];
}

}  // namespace detail

}  // namespace ifgf::interp
