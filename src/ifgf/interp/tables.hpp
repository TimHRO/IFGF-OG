#pragma once

// Host-side Chebyshev tables, computed in double and stored as Real.

#include <ifgf/interp/orders.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ifgf::interp {

// i-th Chebyshev node of order n: the zeros of T_n, cos((2i+1)pi/(2n)), descending in i.
inline double chebNode(int i, int n) { return std::cos((2 * i + 1) * M_PI / (2.0 * n)); }

// Matrix of the values -> coefficients transform for one axis of order n, row-major:
//   c_j = sum_i M[j*n + i] f_i,   M[j][i] = (2 - delta_j0)/n * T_j(x_i) = (2 - delta_j0)/n * cos(j(2i+1)pi/(2n))
template <class Real>
std::vector<Real> transformMatrix(int n)
{
  std::vector<Real> m(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      m[j * n + i] = static_cast<Real>((j == 0 ? 1.0 : 2.0) / n * std::cos(j * (2 * i + 1) * M_PI / (2.0 * n)));
  return m;
}

// Transform matrices of the three axes, concatenated: axis d starts at offset(d).
template <class Real>
struct TransformTables {
  Orders orders;
  std::vector<Real> data;

  explicit TransformTables(const Orders& o) : orders(o)
  {
    if (!o.valid()) throw std::invalid_argument("ifgf::interp: order out of range [1, maxOrder]");
    for (int d = 0; d < 3; ++d) {
      const auto m = transformMatrix<Real>(o.n[d]);
      data.insert(data.end(), m.begin(), m.end());
    }
  }
  std::size_t offset(int axis) const
  {
    std::size_t off = 0;
    for (int d = 0; d < axis; ++d) off += static_cast<std::size_t>(orders.n[d]) * orders.n[d];
    return off;
  }
  const Real* axis(int d) const { return data.data() + offset(d); }
};

}  // namespace ifgf::interp
