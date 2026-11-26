# IFGF-OG
### Interpolated Factored Green Function on GPUs

IFGF-OG is a GPU-accelerated implementation of the **Interpolated Factored Green Function (IFGF)** method, originally introduced in:

Christoph Bauinger and Oscar P. Bruno, “Interpolated Factored Green Function” Method for Accelerated Solution of Scattering Problems, *Journal of Computational Physics*, Volume 430, 2021.

The IFGF method provides an alternative to the Fast Multipole Method (FMM). Rather than relying on multipole expansions, it employs hierarchical Chebyshev interpolation of factored Green's functions to efficiently evaluate boundary integral operators, achieving an asymptotic computational complexity of O(N log N).

This is the first public version (0.1.0). It contains the engine, an example that you can run on your own GPU (`face_test`) and an installable C++ library.

## Key Features

- **Boundary integral operators:** Single-layer (SL), double-layer (DL), combined-field (CF) and hypersingular (HS) operators for scalar and vector-valued densities, including complex wavenumbers
- **High-frequency stability:** Demonstrated stability at high frequencies, with the largest test case reaching κD = 1000
- **Adaptive geometry handling:** Efficient treatment of inhomogeneous geometries using an adaptive octree
- **Interpolation in pyramidal frusta:** around every box, space is split into six pyramids over the faces of a cube and subdivided into smaller cells.
- **Multi-wavenumber evaluation (in preparation):** evaluation of operators for multiple complex wavenumbers using a single initialization, for Convolution Quadrature and Model Order Reduction. The library API already accepts the wavenumber range of an initialization (`maxk`, `minSigma`, `tolerance`); in 0.1.0 these parameters have no effect yet

## Requirements

- A C++20 compiler (clang) and CMake ≥ 3.24
- [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) (SYCL), built with the same clang that compiles IFGF-OG (otherwise the C++ standard libraries of the two do not match at link time)
- For NVIDIA GPUs: CUDA, and AdaptiveCpp built with its CUDA backend

Tested with AdaptiveCpp 25.10, LLVM 20/21 and CUDA 12.8 on an NVIDIA A40.

## Build

Choose the AdaptiveCpp target with a preset (or with `-DIFGF_ACPP_TARGETS=...`):

| preset | target | use |
|---|---|---|
| `cuda-sm86` | `cuda:sm_86` | ahead-of-time CUDA for A40 / RTX 30xx (fastest) |
| `cuda-sm80`, `cuda-sm90` | `cuda:sm_80`, `cuda:sm_90` | A100; H100 / H200 |
| `generic` | `generic` | one binary for CPUs and GPUs; kernels are JIT-compiled at the first apply |
| `cpu` | `omp.library-only` | CPU only |

```bash
cmake --preset cuda-sm86 -DCMAKE_CXX_COMPILER=/path/to/clang++ -DAdaptiveCpp_DIR=/path/to/adaptivecpp/lib/cmake/AdaptiveCpp
cmake --build build/cuda-sm86 -j
```

Options: `IFGF_BUILD_EXAMPLES` (on), `IFGF_BUILD_LIBRARY` (on), `IFGF_NATIVE` (off; `-march=native` for the host code).

## Run the example

`face_test` evaluates one operator on N random points on the unit sphere and compares the result with the exact (FP64) sum at 200 random points:

```bash
./build/cuda-sm86/face_test 400000 25 0 SL 300          # 400k points, kappa = 25, single layer, 300 points per leaf
./build/cuda-sm86/face_test 1572864 300 0 SL 300        # 1.57M points, kappa = 300 (kappa D = 600), ~16 GB
```

Arguments: `face_test [N] [kappa.re] [kappa.im] [SL|DL|CF] [leafSize] [C] [delta] [reps] [fu] [fw] [P]`, see the top of `examples/face_test/face_test.cpp`. It prints the setup time, the device memory, the time of each stage of an apply (near field, leaf, upward, far field) and the relative l2 error.

Expected on an NVIDIA A40 (`cuda:sm_86`, single layer, P = 8, leaf size 300, best of 3 applies):

| N | κ | κD | apply | near | leaf | upward | far | rel. l2 error |
|---|---|---|---|---|---|---|---|---|
| 100,000 | 4π | 25 | 19.8 ms | 1.6 | 7.0 | 3.6 | 7.0 | 6.9e-7 |
| 400,000 | 25 | 50 | 78.4 ms | 4.4 | 24.6 | 16.2 | 31.1 | 8.1e-7 |
| 2,000,000 | 250 | 500 | 656 ms | 23.9 | 159.1 | 232.0 | 225.1 | 1.7e-6 |
| 1,572,864 | 402 | 804 | 651 ms | 50.6 | 276.7 | 175.5 | 135.0 | 6.7e-6 |


The double layer and the combined field take 696 ms at κ = 300 (errors 5.8e-6 and 1.5e-6). At κ = 300 the setup takes about 1.4 s and the operator holds about 16 GB of device memory.

## Use the library

The shared library `ifgf_library` has a plain C++17 interface (`<ifgf/ifgf_library.hpp>`, no SYCL needed by the caller):

```bash
cmake --install build/cuda-sm86 --prefix $HOME/ifgf
cmake -S examples/api_consumer -B build/api_consumer -DCMAKE_PREFIX_PATH=$HOME/ifgf
cmake --build build/api_consumer && ./build/api_consumer/api_consumer 50000 10
```

```cpp
#include <ifgf/ifgf_library.hpp>

ifgf::Options opt;                                     // leaf size, order, cone density, device, ...
ifgf::Operator op(ifgf::Kernel::DoubleLayer, kappa, 3, opt);   // 3 components per point
op.init(src, numSrc, tgt, numTgt, srcNormals);         // xyz interleaved doubles
op.mult(w, 3 * numSrc, u, 3 * numTgt);                 // complex<float>, point-major w[s * 3 + c]
```

In CMake: `find_package(ifgf CONFIG REQUIRED)` and `target_link_libraries(<target> PRIVATE ifgf::ifgf_library)`. The kernels, parameters and data layouts are documented in the header.

Environment variables: `IFGF_DEVICE` = `gpu` (default; the CPU if there is no GPU) | `cpu` | device index; `IFGF_VERBOSE=1` prints setup and apply timings.

## Limitations of 0.1.0

- FP32 evaluation (with compensated geometry for the near field and the far-field phase); relative errors around 1e-6
- One GPU per operator
- Device memory grows with the number of cones (about 3 KB per cone at P = 8; 1.57M points at κ = 300 need about 16 GB)
- The wavenumber-range parameters for multi-wavenumber evaluation are accepted but have no effect yet

## Future Work

- Parallelization across multiple GPUs
- Support for Apple Metal
- FP64 implementation to achieve higher numerical accuracy beyond FP32 precision
- Fused evaluation of vector densities (one pass for all components)
.
