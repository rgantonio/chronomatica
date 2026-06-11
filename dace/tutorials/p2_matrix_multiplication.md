# DaCe Mastery Tutorial
## Part 2: Matrix Multiplication — 2D Maps, WCR, and LibraryNodes

> **Prerequisites:** Complete Part 1. You should be comfortable reading SDFGs
> visually, inspecting them programmatically, and building simple SDFGs by hand.

---

## 2.1 The Matrix Multiplication Operation

Matrix multiplication $C = A \times B$ where $A \in \mathbb{R}^{M \times K}$,
$B \in \mathbb{R}^{K \times N}$, $C \in \mathbb{R}^{M \times N}$ is:

$$C[i,j] = \sum_{k=0}^{K-1} A[i,k] \cdot B[k,j]$$

This introduces three new SDFG concepts not present in AXPY:
- **2D maps** — iterating over two indices simultaneously
- **WCR (Write Conflict Resolution)** — accumulating partial results safely
- **LibraryNodes** — abstract operations with swappable implementations

---

## 2.2 The LibraryNode Surprise

A natural first attempt is to auto-generate the matmul SDFG directly:

```python
import dace
import numpy as np

M, N, K = dace.symbol('M'), dace.symbol('N'), dace.symbol('K')

@dace.program
def matmul(A: dace.float64[M, K], B: dace.float64[K, N]):
    return A @ B

sdfg = matmul.to_sdfg()
sdfg.simplify()
sdfg.save('matmul_auto.sdfg')
sdfg.view()
```

---

### Exercise 2.2: Reading the Auto-Generated Matmul SDFG

Open `matmul_auto.sdfg` and answer:

**Q1.** Do you see maps and tasklets, or something different?

<details>
<summary>Expected answer</summary>

You see a **LibraryNode** — a hexagon or document looking block labeled `MatMul` — not explicit maps and tasklets. DaCe recognizes the `@` operator as a high-level linear algebra operation and keeps it abstract rather than expanding it into loops.

This is intentional. LibraryNodes are abstract operations with multiple possible
**implementations** that can be swapped out:
- `pure` — expanded to explicit maps and tasklets
- `BLAS` — calls cuBLAS/CBLAS
- Your custom accelerator implementation

This is exactly the mechanism you'll use for accelerator design space exploration.

</details>

---

**Q2.** How do you expose the internal map structure?

<details>
<summary>Expected answer</summary>

Call `sdfg.expand_library_nodes()` to replace every LibraryNode with its
concrete implementation:

```python
sdfg = matmul.to_sdfg()
sdfg.simplify()
sdfg.expand_library_nodes()   # ← expand before saving
sdfg.save('matmul_auto.sdfg')
sdfg.view()
```

After expansion you'll see the full map/tasklet structure including the init
state and the 3D gemm map.

</details>

---

**Q3.** After expanding, how many states do you see and what does each one do?

<details>
<summary>Expected answer</summary>

Two states inside the nested SDFG:

- `_MatMult_gemm_initstate` — initializes output `_c` to zero using a
  mapped tasklet over `[0:M, 0:N]`. Required because the accumulation
  starts from zero.
- `_MatMult_gemm_state` — the actual multiplication using a 3D map over
  `[0:M, 0:N, 0:K]` with a WCR reduction.

The init state exists because GEMM is the general operation
$C = \alpha AB + \beta C$. When $\beta = 0$ (pure matmul), C must be zeroed
before accumulation begins.

</details>

---

**Q4.** The memlet volume on `_a` going into the 3D map is $M \times K \times N$,
not just $M \times K$. Why?

<details>
<summary>Expected answer</summary>

The map iterates over all three dimensions `[M, N, K]`. For every output element
$C[i,j]$, you read `A[i,k]` — but you do this for all $j$ values. So `A[i,k]`
gets read $N$ times total across all map iterations.

DaCe computes the **total data volume** across all map iterations:
$$\text{volume}(A) = M \times K \times N$$

This is precisely what the roofline model needs — not the array size, but the
total bytes actually transferred during execution. The MKN volume in the SDFG
is DaCe already doing part of your roofline accounting.

</details>

---

**Q5.** What is the broken arrow on the output edge of the gemm tasklet?

<details>
<summary>Expected answer</summary>

The broken/dashed arrow indicates a **Write Conflict Resolution (WCR)** memlet.
The 3D map iterates over $[i, j, k]$ simultaneously, so multiple iterations
write to the same $C[i,j]$ — once for each value of $k$. This is a reduction.

The WCR annotation `lambda x, y: x + y` tells DaCe: when multiple iterations
write to the same location, resolve the conflict by **adding** them together.
DaCe uses this to generate correct parallel reduction code — without it,
parallel writes to the same cell would be a race condition.

</details>

---

## 2.3 2D Memlet Syntax

Before building matmul manually, you need 2D memlet syntax. It extends naturally
from 1D — just add more indices separated by commas:

```python
# Full 2D array — outside the map
dace.Memlet('A[0:M, 0:K]')

# Single element at iteration i, k — inside the map
dace.Memlet('A[i, k]')

# A row — all columns for a fixed row i
dace.Memlet('A[i, 0:K]')

# Explicit form
dace.Memlet(data='A', subset='i, k')
dace.Memlet(data='A', subset='0:M, 0:K')
```

**WCR memlet syntax:**

```python
dace.Memlet(data='C', subset='i, j', wcr='lambda x, y: x + y')
```

**The scoping rule for nested maps:**

Each time you cross a map boundary inward, one more index gets bound and the
subset shrinks:

| Position | A subset | B subset |
|---|---|---|
| Outside both maps | `A[0:M, 0:K]` | `B[0:K, 0:N]` |
| Inside outer map only | `A[i, 0:K]` | `B[0:K, j]` |
| Inside both maps | `A[i, k]` | `B[k, j]` |

---

## 2.4 Connector Naming and Namespacing

Before building matmul, one important rule: connector names must be **unique
per node** but can be reused across different nodes.

```
outer_entry:  IN_A, IN_B  (its own namespace)
inner_entry:  IN_A, IN_B  (separate namespace — no conflict)
mac_tasklet:  _a, _b      (tasklet connector names match code variables)
```

The convention is `IN_<arrayname>` for inputs and `OUT_<arrayname>` for outputs.

---

## 2.5 Building Matmul by Hand

```python
import dace
import numpy as np

M, N, K = dace.symbol('M'), dace.symbol('N'), dace.symbol('K')
sdfg = dace.SDFG('matmul_manual')

# Arrays
sdfg.add_array('A', shape=[M, K], dtype=dace.float64)
sdfg.add_array('B', shape=[K, N], dtype=dace.float64)
sdfg.add_array('C', shape=[M, N], dtype=dace.float64)

state = sdfg.add_state('compute')

# Access nodes
A_node = state.add_read('A')
B_node = state.add_read('B')
C_node = state.add_write('C')

# Outer map: iterates over output elements [i, j]
outer_entry, outer_exit = state.add_map(
    'outer', {'i': '0:M', 'j': '0:N'})

# Inner map: reduction over k
inner_entry, inner_exit = state.add_map(
    'inner', {'k': '0:K'})

# Tasklet: multiply (addition handled by WCR)
mac_tasklet = state.add_tasklet(
    name='mac',
    inputs={'_a', '_b'},
    outputs={'_tmp'},
    code='_tmp = _a * _b'
)

# Declare connectors
outer_entry.add_in_connector('IN_A')
outer_entry.add_in_connector('IN_B')
outer_entry.add_out_connector('OUT_A')
outer_entry.add_out_connector('OUT_B')

inner_entry.add_in_connector('IN_A')
inner_entry.add_in_connector('IN_B')
inner_entry.add_out_connector('OUT_A')
inner_entry.add_out_connector('OUT_B')

inner_exit.add_in_connector('IN_C')
inner_exit.add_out_connector('OUT_C')

outer_exit.add_in_connector('IN_C')
outer_exit.add_out_connector('OUT_C')

# Input edges — applying the scoping rule
state.add_edge(A_node, None, outer_entry, 'IN_A',
               dace.Memlet('A[0:M, 0:K]'))
state.add_edge(B_node, None, outer_entry, 'IN_B',
               dace.Memlet('B[0:K, 0:N]'))
state.add_edge(outer_entry, 'OUT_A', inner_entry, 'IN_A',
               dace.Memlet('A[i, 0:K]'))
state.add_edge(outer_entry, 'OUT_B', inner_entry, 'IN_B',
               dace.Memlet('B[0:K, j]'))
state.add_edge(inner_entry, 'OUT_A', mac_tasklet, '_a',
               dace.Memlet('A[i, k]'))
state.add_edge(inner_entry, 'OUT_B', mac_tasklet, '_b',
               dace.Memlet('B[k, j]'))

# Output edges — WCR on the tasklet output
state.add_edge(mac_tasklet, '_tmp', inner_exit, 'IN_C',
               dace.Memlet(data='C', subset='i, j',
                           wcr='lambda x, y: x + y'))
state.add_edge(inner_exit, 'OUT_C', outer_exit, 'IN_C',
               dace.Memlet('C[i, j]'))
state.add_edge(outer_exit, 'OUT_C', C_node, None,
               dace.Memlet('C[0:M, 0:N]'))

sdfg.validate()
sdfg.save('matmul_manual.sdfg')

compiled = sdfg.compile()
A = np.random.rand(64, 32)
B = np.random.rand(32, 64)
C = np.zeros((64, 64))
compiled(A=A, B=B, C=C, M=64, N=64, K=32)

expected = A @ B
print("Max error:", np.max(np.abs(C - expected)))
```

### Key concepts in this implementation

**Why WCR instead of an explicit accumulator?**

The WCR on the output memlet handles the reduction implicitly:

```
tasklet output:  '_tmp = _a * _b'     (multiply — 1 op)
WCR on memlet:   'lambda x, y: x + y' (accumulate — 1 op)
```

Where `x` is the **existing value** already in `C[i,j]` and `y` is the
**new contribution** from this iteration. The WCR is the addition — you never
need a separate add tasklet or explicit accumulator variable.

**Why the WCR is visible as a broken arrow:**

Multiple $k$ iterations all write to the same `C[i,j]`. Without WCR this would
be a race condition. The broken arrow in the viewer signals that DaCe will
generate correct parallel reduction code for this edge.

---

### Exercise 2.5: Manual Matmul

Build and run the manual matmul above, then answer:

**Q1.** Open `matmul_manual.sdfg` and `matmul_auto.sdfg` side by side. The
auto-generated version uses a single 3D map while yours uses nested 2D+1D maps.
Are both correct?

<details>
<summary>Expected answer</summary>

Yes — both are correct representations of the same computation. They differ in
how the parallelism is structured:

- **Auto-generated (3D map):** All three loops fused into one parallel scope.
  DaCe can schedule all $M \times N \times K$ iterations simultaneously.
- **Manual (nested 2D+1D):** Outer scope handles output elements, inner scope
  handles the reduction. More explicit about the reduction structure.

The auto-generated version comes from the GEMM library expansion which uses
`add_mapped_tasklet` to build a 3D map directly. Both produce identical results.

</details>

---

**Q2.** What does the WCR `lambda x, y: x + y` mean — what are `x` and `y`?

<details>
<summary>Expected answer</summary>

- `x` = the **existing value** already stored at `C[i,j]` (the running sum)
- `y` = the **new contribution** arriving from the tasklet output this iteration

DaCe generates atomic or reduction code to safely compute `C[i,j] = x + y`
even when multiple parallel iterations write to the same location. The WCR
replaces what would otherwise be a race condition with correct accumulation.

</details>

---

**Q3.** Why does the memlet subset change as you cross each map boundary inward?

<details>
<summary>Expected answer</summary>

Each map boundary binds one more index variable. The subset must reflect what
data is **visible at that scope**:

- Outside `outer`: `i` and `j` are unbound → full arrays `A[0:M, 0:K]`
- Inside `outer`, outside `inner`: `i` and `j` are bound, `k` is not
  → `A[i, 0:K]` (one row of A for this `i`)
- Inside both: `i`, `j`, and `k` are all bound → `A[i, k]` (one element)

This two-level annotation tells DaCe both the **total data movement** across
the full map and the **per-iteration access pattern** — both are needed for
correct code generation and roofline analysis.

</details>

---

## 2.6 Reading the GEMM Library Source

DaCe's GEMM expansion is in `dace/libraries/blas/nodes/matmul.py`.
The key section is `ExpandGemmPure.make_sdfg`. Understanding it connects
everything you've learned:

```python
# The 3D multiplication map
state.add_mapped_tasklet(
    "gemm",
    {"__i%d" % i: "0:%s" % s for i, s in enumerate([M, N, K])},
    {
        "__a": dace.Memlet.simple("_a",
               "__i2, __i0" if node.transA else "__i0, __i2"),
        "__b": dace.Memlet.simple("_b",
               "__i1, __i2" if node.transB else "__i2, __i1")
    },
    mul_program,   # '__out = __a * __b'
    {"__out": dace.Memlet.simple(
        mul_out, "__i0, __i1",
        wcr_str="lambda x, y: x + y")},  # ← WCR here
    external_edges=True
)
```

| What you saw in viewer | Where it comes from in code |
|---|---|
| Init state zeroing `_c` | `equal_valued(0, node.beta)` branch |
| 3D map `i0:M, i1:N, i2:K` | `enumerate([M, N, K])` dict comprehension |
| `_a` indexed as `__i0, __i2` | `"__i2, __i0" if node.transA else "__i0, __i2"` |
| Broken arrow WCR | `wcr_str="lambda x, y: x + y"` |
| MKN total volume | Total iterations × per-iteration access |

---

## 2.7 SDFG Reading Without a Viewer

When analyzing large SDFGs programmatically, you need to read the graph structure
from code. Use this pattern:

```python
import dace

sdfg = dace.SDFG.from_file('matmul_auto.sdfg')

def find_all_maps(sdfg, indent=0):
    """Recursively find all maps including inside nested SDFGs."""
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                print(' ' * indent +
                      f"Map: {node.label}  "
                      f"range: {node.map.range}  "
                      f"in state: {state.label}")
            if isinstance(node, dace.nodes.NestedSDFG):
                print(' ' * indent + f"[NestedSDFG: {node.label}]")
                find_all_maps(node.sdfg, indent + 4)

find_all_maps(sdfg)
```

> **Important:** Maps inside LibraryNode expansions live inside NestedSDFGs,
> one level deeper than the top-level state. Always recurse into NestedSDFGs
> when searching for maps. The expanded matmul SDFG will show:
>
> ```
> [NestedSDFG: _MatMult_gemm]
>     Map: gemm_init_map  range: 0:M, 0:N  in state: _MatMult_gemm_initstate
>     Map: gemm_map  range: 0:M, 0:N, 0:K  in state: _MatMult_gemm_state
> ```

---

## Summary: Part 2

You now understand:

- The `@` operator generates a **LibraryNode**, not explicit maps — use
  `expand_library_nodes()` to expose the internal structure
- **2D memlets** use comma-separated indices: `A[i, k]`, `C[0:M, 0:N]`
- **Nested maps** require the scoping rule: each boundary binds one more index
- **WCR** (`lambda x, y: x + y`) handles reductions — the broken arrow in the
  viewer signals a write conflict that DaCe resolves safely
- **Connector namespacing** is per-node — `IN_A` on `outer_entry` and `IN_A`
  on `inner_entry` are independent
- Maps inside LibraryNode expansions live inside **NestedSDFGs** — always
  recurse when searching programmatically
- The MKN memlet volume reflects **total data moved** across all iterations,
  not just array size — this is the number roofline analysis needs

**Key insight for your methodology:** The GEMM at arithmetic intensity
$I = 8.0$ FLOPs/byte (for $M=N=512$, $K=256$) is firmly compute-bound.
This is directly readable from the SDFG structure — the 3D map range and
the per-iteration memory access pattern give you everything you need.

**Next:** Part 3 covers transformations — MapFusion, MapTiling, and surgical
targeting of specific subgraphs.
