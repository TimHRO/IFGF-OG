#pragma once

// Device selection by name.

#include <sycl/sycl.hpp>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ifgf::exec {

// "cpu", "gpu", or a device index into sycl::device::get_devices().
inline sycl::device selectDevice(std::string_view name)
{
  if (name == "cpu") return sycl::device{sycl::cpu_selector_v};
  if (name == "gpu") return sycl::device{sycl::gpu_selector_v};

  const std::string s{name};
  std::size_t pos = 0;
  int index = -1;
  try {
    index = std::stoi(s, &pos);
  } catch (const std::exception&) {
    pos = 0;
  }
  if (pos != s.size() || index < 0)
    throw std::invalid_argument("ifgf: unknown device '" + s + "' (use cpu, gpu or an index)");

  const auto devices = sycl::device::get_devices();
  if (static_cast<std::size_t>(index) >= devices.size())
    throw std::out_of_range("ifgf: device index " + s + " out of range");
  return devices[static_cast<std::size_t>(index)];
}

// Device named by the environment variable `var`, falling back to `fallback`.
inline sycl::device deviceFromEnv(const char* var = "IFGF_DEVICE", std::string_view fallback = "cpu")
{
  const char* v = std::getenv(var);
  return selectDevice((v && *v) ? std::string_view{v} : fallback);
}

inline std::string describe(const sycl::device& d)
{
  return d.get_info<sycl::info::device::name>() + " [" +
         d.get_platform().get_info<sycl::info::platform::name>() + "]";
}

}  // namespace ifgf::exec
