#pragma once

// Direct-sum reference evaluator (host): u(t) = sum_s w_s K(t, s), the ground truth for
// accuracy checks. Instantiate it with a double-precision kernel. Parallel over
// targets with std::thread; the order of summation per target is fixed, so the result is
// deterministic.

#include <ifgf/core/vec3.hpp>

#include <algorithm>
#include <span>
#include <thread>
#include <vector>

namespace ifgf::kernels {

template <class Kernel>
std::vector<typename Kernel::Value> directSum(const Kernel& k, std::span<const Vec3d> sources,
                                              std::span<const typename Kernel::SrcData> srcData,
                                              std::span<const typename Kernel::Value> weights,
                                              std::span<const Vec3d> targets, unsigned numThreads = 0)
{
  using Value = typename Kernel::Value;
  std::vector<Value> out(targets.size(), Value(0));
  if (numThreads == 0) numThreads = std::max(1u, std::thread::hardware_concurrency());
  numThreads = std::min<unsigned>(numThreads, static_cast<unsigned>(std::max<std::size_t>(targets.size(), 1)));
  auto work = [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      Value sum(0);
      for (std::size_t j = 0; j < sources.size(); ++j) {
        const typename Kernel::SrcData d = srcData.empty() ? typename Kernel::SrcData{} : srcData[j];
        sum += weights[j] * k.direct(targets[i], sources[j], d);
      }
      out[i] = sum;
    }
  };
  std::vector<std::thread> pool;
  const std::size_t chunk = (targets.size() + numThreads - 1) / numThreads;
  for (unsigned t = 0; t < numThreads; ++t) {
    const std::size_t b = t * chunk, e = std::min(targets.size(), b + chunk);
    if (b < e) pool.emplace_back(work, b, e);
  }
  for (auto& th : pool) th.join();
  return out;
}

}  // namespace ifgf::kernels
