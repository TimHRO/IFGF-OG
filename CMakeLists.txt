cmake_minimum_required(VERSION 3.24)

project(ifgf VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Options -----------------------------------------------------------------------------------------------------
# AdaptiveCpp compilation targets (ACPP_TARGETS), for example
#   generic           single-pass compiler: one binary for CPUs and GPUs, kernels JIT-compiled at the first apply
#   cuda:sm_86        ahead-of-time CUDA for one architecture (A40/RTX 30xx: sm_86, A100: sm_80, H100: sm_90)
#   omp.library-only  CPU only, through OpenMP
set(IFGF_ACPP_TARGETS "generic" CACHE STRING "AdaptiveCpp targets (ACPP_TARGETS)")
option(IFGF_BUILD_LIBRARY "Build the installable shared library ifgf_library" ON)
option(IFGF_BUILD_EXAMPLES "Build the face_test example" ON)
option(IFGF_NATIVE "Optimise the host code for the build machine (-march=native)" OFF)
# CUDA: AdaptiveCpp's kernel entry stub calls the kernel through small wrappers. With LLVM's default inline threshold a
# large kernel body is not inlined into them and every thread copies the kernel's captures to its stack. A high
# device-only inline threshold avoids that (about 13 % faster applies).
set(IFGF_GPU_INLINE_THRESHOLD 20000 CACHE STRING "Device inline threshold for cuda:* targets (0: LLVM default)")

# AdaptiveCpp reads ACPP_TARGETS during find_package.
set(ACPP_TARGETS "${IFGF_ACPP_TARGETS}" CACHE STRING "AdaptiveCpp targets" FORCE)
find_package(AdaptiveCpp CONFIG REQUIRED)
message(STATUS "ifgf: AdaptiveCpp targets = ${ACPP_TARGETS}")

# --- Common flags ------------------------------------------------------------------------------------------------
add_library(ifgf_options INTERFACE)
target_compile_options(ifgf_options INTERFACE
  -Wall -Wextra -Wpedantic
  "$<$<CONFIG:Release>:-O3>"
  "$<$<CONFIG:RelWithDebInfo>:-O2;-g>"
  "$<$<CONFIG:Debug>:-O0;-g>"
)
if(IFGF_NATIVE)
  target_compile_options(ifgf_options INTERFACE -march=native)
endif()
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(WARNING "ifgf: the C++ compiler is ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER}). Use the clang that "
                  "AdaptiveCpp was built with (-DCMAKE_CXX_COMPILER=...), otherwise linking usually fails.")
endif()
if(IFGF_ACPP_TARGETS MATCHES "cuda:" AND IFGF_GPU_INLINE_THRESHOLD GREATER 0 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(ifgf_options INTERFACE
    "-fgpu-inline-threshold=${IFGF_GPU_INLINE_THRESHOLD}" -Wno-unused-command-line-argument)
endif()

include(GNUInstallDirs)

add_subdirectory(src/ifgf)

if(IFGF_BUILD_EXAMPLES)
  add_executable(face_test examples/face_test/face_test.cpp)
  target_link_libraries(face_test PRIVATE ifgf::ifgf)
  add_sycl_to_target(TARGET face_test SOURCES examples/face_test/face_test.cpp)
  target_compile_options(face_test PRIVATE -Wno-pass-failed)
endif()
