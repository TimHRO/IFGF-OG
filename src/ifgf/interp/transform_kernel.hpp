#pragma once

// Values -> coefficients inside a work-group. SYCL.
//
// transformInGroup is called by the leaf kernel after it has written a cone's node values into local memory:
// three in-place passes on that one buffer, split over the group's threads, with a barrier between passes.

#include <ifgf/interp/orders.hpp>
#include <ifgf/interp/transform.hpp>

#include <sycl/sycl.hpp>


namespace ifgf::interp {

// Transform matrices of the three axes (TransformTables::data on the device).
template <class Real>
struct DeviceTransformTables {
  const Real* M[3];
};

// data: the cone's node values in local memory, [k][W]. Precondition: all values are
// written and visible (the caller has placed a barrier). Postcondition: data holds the
// coefficients, [a][W], visible to all threads of the group.
// N0, N12: the radial order and the (equal) angular orders as compile-time constants, so the
// passes are fully unrolled without guards; 0 = runtime orders.
template <class Real, class Acc = Real, int N0 = 0, int N12 = 0>
inline void transformInGroup(const sycl::nd_item<1>& it, Real* data, const Orders& o,
                             const DeviceTransformTables<Real>& tab, int W)
{
  const int tid = static_cast<int>(it.get_local_id(0));
  const int nt = static_cast<int>(it.get_local_range(0));
  transformPassN<Real, Acc, N0>(data, o, 0, tab.M[0], W, tid, nt);
  sycl::group_barrier(it.get_group());
  transformPassN<Real, Acc, N12>(data, o, 1, tab.M[1], W, tid, nt);
  sycl::group_barrier(it.get_group());
  transformPassN<Real, Acc, N12>(data, o, 2, tab.M[2], W, tid, nt);
  sycl::group_barrier(it.get_group());
}

}  // namespace ifgf::interp
