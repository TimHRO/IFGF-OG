# IFGF-OG
### Interpolated Factored Green Function Evaluation on GPUs

IFGF-OG is a GPU-accelerated implementation of the **Interpolated Factored Green Function (IFGF)** method, originally introduced in:

Christoph Bauinger and Oscar P. Bruno, “Interpolated Factored Green Function” Method for Accelerated Solution of Scattering Problems, *Journal of Computational Physics*, Volume 430, 2021.

The IFGF method provides an alternative to the Fast Multipole Method (FMM). Rather than relying on multipole expansions, it employs hierarchical Chebyshev interpolation of factored Green's functions to efficiently evaluate boundary integral operators, achieving an asymptotic computational complexity of O(N log N).

## Key Features

- **Boundary integral operators:** Single-layer (SL), double-layer (DL), and hypersingular (HS) operators for scalar and vector-valued densities, including complex wavenumbers
- **High-frequency stability:** Demonstrated stability at high frequencies, with the largest test case reaching κD = 1000
- **Adaptive geometry handling:** Efficient treatment of inhomogeneous geometries using an adaptive octree
- **Multi-wavenumber evaluation:** Evaluation of operators for multiple complex wavenumbers using a single initialization. This makes IFGF-OG particularly well suited for applications involving Convolution Quadrature and Model Order Reduction, where repeated operator evaluations across sets of complex wavenumbers are required

## Future Work

- Parallelization across multiple GPUs
- Support for Apple Metal
- FP64 implementation to achieve higher numerical accuracy beyond FP32 precision
