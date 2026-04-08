## Naming conventions for 3D mesh edges

For a given tetrahedron `ijkl`. We define a "local reference frame" for each edge. For example, consider the edge `ij` as the "self", or `s`. Then

* the opposite edge, `kl`, can be denoted as `o`.
* The edge `ik` connecting the tail of `ij` and `kl` can be denoted as `tt`.
* The edge `jl` connecting the head of `ij` and `kl` can be denoted as `hh`.
* The edge `il` connecting the tail of `ij` with head of `kl` is `th`.
* The edge `jk` connecting the head of `ij` with tail of `kl` is `ht`.

These local relations can be translated into global relations as follows (using
canonical edge orientations):

| s | o | tt | hh | th | ht |
| - | - | - | - | - | - |
| ij | kl | ik | jl | il | jk |
| ik | jl | ij | kl | il | jk |
| jk | il | ij | kl | jl | ik |
| jl | ik | ij | kl | jk | il |
| kl | ij | ik | jl | jk | il |
| il | jk | jl | ik | kl | ij |

In addition, we tabulate the following jacobians outward facing normal vectors and their Jacobians. Here, the `[]` notation maps a vector `t` to a skew symmetric matrix `[t]`, such that `t x v = [t]v` for all vectors `v`.

First, we tabulate the th x o normal vector (i.e., the double area normal that is outward facing if the tet is positively oriented) and its gradient wrt vertex coordinates. Note that, in the `th x o` column, the cross product in the parenthesis is rearranged to be outward facing and "corner-centered", and this is the version implemented in the code.

| s | th x o  (-> oriented) | grad_i | grad_j | grad_k | grad_l |
| - | - | - | - | - | - |
| ij | il x kl (-> lk x li) | [lk]=[-o] | [ll]=0 | [il]=[th] | [ki]=[-tt] |
| ik | il x jl (-> li x lj) | [jl]=[o] | [li]=[-th] | [ll]=0 | [ij]=[tt] |
| jk | jl x il (-> li x lj) | [jl]=[th] | [li]=[-o] | [ll]=0 | [ij]=[tt] |
| jl | jk x ik (-> kj x ki) | [kj]=[-th] | [ik]=[o] | [ji]=[-tt] | [kk]=0 |
| kl | jk x ij (-> ji x jk) | [kj]=[-th] | [ik]=[tt] | [ji]=[-o] | [jj]=0 |
| il | kl x jk (-> kl x kj) | [kk]=0 | [kl]=[th] | [lj]=[-tt] | [jk]=[o] |

Then, we tabulate the `hh x o` normal vector and its gradient wrt vertex coordinates:

| s | hh x o  (-> oriented) | grad_i | grad_j | grad_k | grad_l |
| - | - | - | - | - | - |
| ij | jl x kl (-> lj x lk) | [ll]=0 | [kl]=[o] | [lj]=[-hh] | [jk]=[ht] |
| ik | kl x jl (-> lj x lk) | [ll]=0 | [kl]=[hh] | [lj]=[-o] | [jk]=[ht] |
| jk | kl x il (-> lk x li) | [lk]=[-hh] | [ll]=0 | [il]=[o] | [ki]=[-ht] |
| jl | kl x ik (-> ki x kl) | [lk]=[-hh] | [kk]=0 | [il]=[ht] | [ki]=[-o] |
| kl | jl x ij (-> jl x ji) | [jl]=[hh] | [li]=[-ht] | [jj]=0 | [ij]=[o] |
| il | ik x jk (-> kj x ki) | [kj]=[-o] | [ik]=[hh] | [ji]=[-ht] | [kk]=0 |