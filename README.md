# Cochain: differentiable operators for computational topology and DEC

**Status**: Pre-alpha, in early development.

Cochain is a collection of computational topology operators built on PyTorch, designed to facilitate the analysis of discrete topological objects (specifically, 2D and 3D simplicial meshes embedded in $\mathbb{R}^3$) within the context of discrete exterior calculus (DEC) and cohomology theory.

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
* Sparse linear algebra utils:
    * PyTorch interfaces for existing sparse linear solvers (SuperLU and cuDSS) and eigensolvers (Lanczos and LOBPCG) that support generalized eigenvalue problems and the shift-invert mode.
    * Autograd support for fixed-topology sparse operations.


### Planned Features:

* Harmonic form generator.
