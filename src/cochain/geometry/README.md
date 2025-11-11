## Naming conventions for 2D mesh vertices

We adopt the following convention for describing the relation between vertices in a triangle. For a given triangle represented by three vertex indices, we refer the first, second, and third vertex with index `i`, `j`, and `k`. This effectively assigns an orientation to the triangle, and allows us to distinguish the neighbors for each vertex ("self", or `s`) as either the "next" (`n`) and "previous" (`p`) vertex. For a triangle `ijk`, the "self"/"next"/"prev" relation is defined as follows:

| s | n | p |
| - | - | - |
| i | j | k |
| j | k | i |
| k | i | j |

We will refer to a triangle as `ijk` or `snp`, depending on the context.