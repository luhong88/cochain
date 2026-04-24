# Cochain: differentiable operators for computational topology and DEC

**Status**: Pre-alpha, in early development.

Cochain is a collection of computational topology operators built on PyTorch, designed to facilitate the analysis of discrete topological objects—specifically, simplicial meshes immersed in $\mathbb R^3$ and their associated discrete cochains—within the context of discrete exterior calculus (DEC) and cohomology theory; the underlying chain complexes are defined over $\mathbb{R}$.

## Features

* Simplicial complexes & combinatorial topology:
    * Piecewise-linear triangular and tetrahedral meshes immersed in $\mathbb{R}^3$.
    * Coboundary operators (discrete exterior derivatives).
    * Reduced coboundary operators via discrete Morse theory.
    * Combinatorial Laplacians on both the primal and dual meshes.
    * Tree-cotree decomposition for 1-Laplacians on triangular meshes.
    * Betti numbers.
* Metric-dependent operators:
    * DEC Hodge stars (circumcentric and barycentric duals) and consistent mass matrices.
    * DEC Hodge Laplacians (for triangular meshes) and weak Laplacians/stiffness matrices (for tetrahedral meshes).
* Cochain operations & mappings:
    * Cup product, anti-symmetrized cup product, and Galerkin ($L^2$-projected) wedge product.
    * Galerkin interior product.
    * Whitney map and de Rham map. 
    * Flat and sharp operators for music isomorphism.
* Sparse linear algebra utils:
    * Block-diagonal mesh batching.
    * PyTorch interfaces for existing sparse linear solvers (SuperLU and cuDSS) and eigensolvers (Lanczos and LOBPCG) that support generalized eigenvalue problems and the shift-invert mode.
    * Autograd support for fixed-topology sparse operations.


### Planned Features

* Harmonic form generator.

## License

This project is licensed under the MIT License; see the `LICENSE` file for details.