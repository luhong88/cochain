# Cochain: a differentiable engine for computational topology and DEC

**Status**: Pre-alpha, in early development.

Cochain is a specialized framework for computational topology written in PyTorch, designed to facilitate the rigorous manipulation of discrete topological objects (specifically, 2D and 3D simplicial meshes embedded in $\mathbb{R}^3$) and to bridge algebraic topology with high-performance scientific computing. By grounding its discrete operators in the formalisms of cohomology theory and discrete exterior calculus, Cochain enables complex topological transformations and structure-preserving discretizations to be evaluated while leveraging the hardware acceleration and automatic differentiation of modern computational backends.

## Features:

* Simplicial complexes & combinatorial topology:
    * Piecewise-linear triangular and tetrahedral meshes embedded in $\mathbb{R}^3$.
    * Coboundary operators (discrete exterior derivatives).
    * Combinatorial Laplacians on both the primal and dual meshes.
    * Tree-cotree decomposition for 1-Laplacians on triangular meshes.
* Metric-dependent operators:
    * Discrete Hodge stars for both circumcentric and barycentric duals.
    * Consistent mass matrices derived from Whitney basis functions (of the lowest order).
    * DEC Hodge Laplacians (for triangular meshes) and weak Laplacians/stiffness matrices (for tetrahedral meshes).
* Cochain operations & mappings:
    * Cup product, anti-symmetrized cup product, and Galerkin ($L^2$-projected) wedge product.
    * Galerkin interior product.
    * Whitney map for interpolation of discrete $k$-cochains and de Rham map for discretization of continuous $k$-forms. 
    * Flat and sharp operators for music isomorphisms between 1-cochains and vector fields.
* Computational backends:
    * Differentiable wrappers for sparse linear solvers SuperLU (via `scipy` and `cupy`) and cuDSS (via `nvmath-python`).
    * Differentiable sparse eigensolvers, including the implicitly restarted Lanczos method (via `scipy`), the thick-restart Lanczos method (via `cupy`), and a custom, GPU-compatible LOBPCG implementation that supports both generalized eigenvalue problems and the shift-invert mode for interior eigenvalues.
    * Fixed-topology autograd: optimized sparse matrix primitives that pre-compute index structures, ensuring efficient backprop through operator values.


### Planned Features:

* Harmonic form generator.
