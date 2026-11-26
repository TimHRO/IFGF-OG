#pragma once

// Interpolation orders per cone.
//
// "Order" is the number of Chebyshev nodes per axis. Axis 0 is the radial variable w,
// axes 1 and 2 are the face-chart coordinates u and v. Node and coefficient index: k = k0 + n0*(k1 + n1*k2),
// axis 0 fastest.

namespace ifgf::interp {

// Capacity of the per-thread register arrays in the device code.
inline constexpr int maxOrder = 8;

struct Orders {
  int n[3] = {1, 1, 1};

  // radial order P-2 (at least 2), angular orders P
  static constexpr Orders fromP(int P) { return Orders{{P - 2 < 2 ? 2 : P - 2, P, P}}; }

  constexpr int nodes() const { return n[0] * n[1] * n[2]; }
  constexpr int stride(int axis) const { return axis == 0 ? 1 : (axis == 1 ? n[0] : n[0] * n[1]); }

  constexpr void split(int k, int out[3]) const
  {
    out[0] = k % n[0];
    out[1] = (k / n[0]) % n[1];
    out[2] = k / (n[0] * n[1]);
  }
  constexpr int flat(int k0, int k1, int k2) const { return k0 + n[0] * (k1 + n[1] * k2); }

  constexpr bool valid() const
  {
    for (int d = 0; d < 3; ++d)
      if (n[d] < 1 || n[d] > maxOrder) return false;
    return true;
  }
};

}  // namespace ifgf::interp
