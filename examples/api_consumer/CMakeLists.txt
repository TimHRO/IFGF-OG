# Consumer of the installed ifgf package, compiled with the plain system compiler (no SYCL), like NGSolve:
#   cmake -S examples/api_consumer -B build/api_consumer -DCMAKE_PREFIX_PATH=<ifgf install prefix>
#   cmake --build build/api_consumer && ./build/api_consumer/api_consumer [N] [kappa.re] [kappa.im]
cmake_minimum_required(VERSION 3.24)
project(ifgf_api_consumer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(ifgf CONFIG REQUIRED)
add_executable(api_consumer main.cpp)
target_link_libraries(api_consumer PRIVATE ifgf::ifgf_library)
