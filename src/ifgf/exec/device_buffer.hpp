#pragma once

// USM device array owned by one queue (all device data is USM).

#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <new>
#include <utility>
#include <vector>

namespace ifgf::exec {

// Process-wide accounting of DeviceBuffer allocations (all queues together): bytes in use and the peak since the
// last resetDeviceBytesPeak() (memory inventory, plans/vector-operators.md §5). Other USM allocations (AdaptiveCpp
// internals, local memory) are not counted.
namespace detail {
inline std::atomic<std::size_t> deviceBytesInUse{0}, deviceBytesPeak{0};
inline void deviceBytesAdd(std::size_t b)
{
  const std::size_t now = deviceBytesInUse.fetch_add(b) + b;
  std::size_t p = deviceBytesPeak.load();
  while (now > p && !deviceBytesPeak.compare_exchange_weak(p, now)) {
  }
}
}  // namespace detail
inline std::size_t deviceBytesInUse() { return detail::deviceBytesInUse.load(); }
inline std::size_t deviceBytesPeak() { return detail::deviceBytesPeak.load(); }
inline void resetDeviceBytesPeak() { detail::deviceBytesPeak.store(detail::deviceBytesInUse.load()); }

template <class T>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(sycl::queue& q, std::size_t n) : q_(&q), n_(n), p_(sycl::malloc_device<T>(std::max<std::size_t>(n, 1), q))
  {
    if (!p_) throw std::bad_alloc();  // malloc_device returns nullptr when the device is out of memory
    detail::deviceBytesAdd(bytes());
  }
  DeviceBuffer(sycl::queue& q, const std::vector<T>& v) : DeviceBuffer(q, v.size()) { upload(v); }
  ~DeviceBuffer() { reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept { swap(o); }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept
  {
    if (this != &o) {
      reset();
      swap(o);
    }
    return *this;
  }

  void upload(const std::vector<T>& v)
  {
    if (!v.empty()) q_->copy(v.data(), p_, std::min(v.size(), n_)).wait();
  }
  std::vector<T> download() const
  {
    std::vector<T> v(n_);
    if (n_) q_->copy(p_, v.data(), n_).wait();
    return v;
  }
  T* data() const { return p_; }
  std::size_t size() const { return n_; }
  std::size_t bytes() const { return p_ ? std::max<std::size_t>(n_, 1) * sizeof(T) : 0; }  // allocated bytes

private:
  void reset()
  {
    if (p_) {
      detail::deviceBytesInUse.fetch_sub(bytes());
      sycl::free(p_, *q_);
    }
    p_ = nullptr;
    n_ = 0;
  }
  void swap(DeviceBuffer& o) noexcept
  {
    std::swap(q_, o.q_);
    std::swap(n_, o.n_);
    std::swap(p_, o.p_);
  }
  sycl::queue* q_ = nullptr;
  std::size_t n_ = 0;
  T* p_ = nullptr;
};

}  // namespace ifgf::exec
