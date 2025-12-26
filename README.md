# Cochain: Differentiable Simplicial Operators in PyTorch

**Status**: Pre-Alpha, in early development.

## Features:

* Simplicial complexes:
    * Piecewise-linear triangular and tetrahedral meshes embedded in $\mathbb{R}^3$.
* Topological operators:
    * Coboundary operators and combinatorial Laplacians.
* Geometric operators:
    * Discrete Hodge stars for both circumcentric and barycentric duals.
    * Mass matrices derived from Whitney basis functions.
    * DEC Hodge Laplacians (for triangular meshes) and weak Laplacians/stiffness matrices (for tetrahedral meshes).
* Computational backend:
    * Differentiable wrappers for sparse linear solvers (SuperLU and cuDSS backends).
    * Fixed-topology autograd: Optimized sparse matrix primitives that pre-compute index structures, ensuring efficient backprop through operator values.


### Planned Features:

* Cup products, antisymmetrized cup products, and Whitney wedge products.
* Differentiable sparse eigensolver wrapper.
* Whitney form interpolation (for point cloud/mesh conversions).
