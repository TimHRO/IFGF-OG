#pragma once

// Values -> coefficients, in place on one buffer. Device-safe.
//
// data: one cone, W real lanes per node (complex values: W = 2, interleaved re, im), layout [k][W].
// One pass per axis: every line along the axis (fixed other two indices, fixed lane) is
// read into registers, multiplied by the n x n transform matrix of that axis, and written
// back in place. Lines are independent, so a pass can be split over threads: thread `tid`
// of `nthreads` takes items tid, tid + nthreads, ...  The caller places a barrier between
// passes.

#include <ifgf/interp/orders.hpp>

namespace ifgf::interp {

// One pass along `axis`. M: n x n matrix of that axis, row-major (row = output coefficient).
// Acc: accumulation type.
// N > 0: the axis order n = N is known at compile time (fully unrolled, no guards);
// N = 0: runtime n <= maxOrder with guards.
template <class Real, class Acc = Real, int N = 0>
inline void transformPassN(Real* data, const Orders& o, int axis, const Real* M, int W, int tid, int nthreads)
{
  constexpr int NI = N > 0 ? N : maxOrder;
  const int n = N > 0 ? N : o.n[axis];
  const int stride = o.stride(axis) * W;
  const int numItems = o.nodes() / n * W;
  for (int item = tid; item < numItems; item += nthreads) {
    const int lane = item % W;  // lane fastest: neighbouring threads touch neighbouring data
    const int line = item / W;
    // first node of the line: insert index 0 for `axis` into the other two indices
    int base;
    if (axis == 0)
      base = n * line;
    else if (axis == 1)
      base = (line % o.n[0]) + o.n[0] * n * (line / o.n[0]);
    else
      base = line;
    Real* p = data + base * W + lane;

    Acc v[NI];
#pragma unroll
    for (int i = 0; i < NI; ++i)
      if (N > 0 || i < n) v[i] = static_cast<Acc>(p[i * stride]);
#pragma unroll
    for (int j = 0; j < NI; ++j)
      if (N > 0 || j < n) {
        Acc s = Acc(0);
#pragma unroll
        for (int i = 0; i < NI; ++i)
          if (N > 0 || i < n) s += static_cast<Acc>(M[j * n + i]) * v[i];
        p[j * stride] = static_cast<Real>(s);
      }
  }
}

}  // namespace ifgf::interp
