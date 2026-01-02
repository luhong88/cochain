# Cochain: Differentiable Simplicial Operators in PyTorch

**Status**: Pre-alpha, in early development.

## Features:

* Simplicial complexes:
    * Piecewise-linear triangular and tetrahedral meshes embedded in $\mathbb{R}^3$.
* Topological operators:
    * Coboundary operators, aka discrete exterior derivatives.
    * Combinatorial Laplacians.
    * Cup product and anti-symmetrized cup product.
* Geometric operators:
    * Discrete Hodge stars for both circumcentric and barycentric duals.
    * Mass matrices derived from Whitney basis functions (of the lowest order).
    * DEC Hodge Laplacians (for triangular meshes) and weak Laplacians/stiffness matrices (for tetrahedral meshes).
    * Galerkin/$L^2$-projected wedge product.
* Computational backend:
    * Differentiable wrappers for sparse linear solvers SuperLU (via `scipy` and `cupy`) and cuDSS (via `nvmath-python`).
    * Differentiable sparse eigensolvers, including the implicitly restarted Lanczos method (via `scipy`), the thick-restart Lanczos method (via `cupy`), and a custom, GPU-compatible LOBPCG implementation that supports both generalized eigenvalue problems and the shift-invert mode for interior eigenvalues.
    * Fixed-topology autograd: optimized sparse matrix primitives that pre-compute index structures, ensuring efficient backprop through operator values.


### Planned Features:

* Musical operators.
* Harmonic form generator.
* Whitney form interpolation (for point cloud/mesh conversions).
