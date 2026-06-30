# Cochain: differentiable operators for computational topology and DEC

Cochain is a collection of computational topology operators built on PyTorch, designed to facilitate the analysis of discrete topological objects—specifically, simplicial meshes immersed in $\mathbb R^3$ and their associated discrete cochains—within the context of discrete exterior calculus (DEC) and cohomology theory; the underlying chain complexes are defined over $\mathbb{R}$.

## Installation

> [!NOTE]
> `cochain` is currently in pre-release and is not yet available on PyPI. For now, please install it directly from GitHub.

First, follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install the correct `PyTorch` version for your OS and compute platform. Then, install the base `cochain` package via `pip`:

```bash
pip install cochain
```

`cochain` is tested against `python>=3.11` and `torch>=2.9.0`, but it will likely work with older versions of both.

### Hardware-accelerated dependencies

Some sparse linear algebra routines require the following additional dependencies to enable CUDA-specific accelerations; currently, `cochain` is tested against CUDA 12.

* `CuPy`: see the [installation guide](https://docs.cupy.dev/en/stable/install.html); version `>=14.0.0` is required for compatibility with `NumPy` 2.0.
* `nvmath-python`: see the [installation guide](https://docs.nvidia.com/cuda/nvmath-python/latest/installation.html#); version `>=0.5.0` is required because earlier versions lack the sparse linear solver utils.

### Optional dependencies

* `vis`: Installs `Polyscope` for visualization of meshes and cochains.
* `examples`: Installs meshing utilities `PyVista` and `PyTetWild`, which are required for generating some example meshes.

These optional dependency groups can be installed using the standard "extras" bracket notation; e.g.,

```bash
pip install cochain[vis,examples]
```

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