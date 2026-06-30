# DaCe Mastery Tutorial
## Part 7: End-to-End Case Study — Self-Attention Analysis

> **Prerequisites:** Complete Parts 1–6. This is the capstone that applies
> everything you have learned to a real transformer component.

---

## 7.1 Why Self-Attention?

Self-attention is the computational core of transformer-based architectures —
JEPA, VLMs, and the neural component of NeSy models all depend on it. It is
also the ideal case study because it combines every concept from this tutorial:

| Component | DaCe concept |
|---|---|
| $QK^T$ matmul | LibraryNode (GEMM) — Part 2 |
| Scale by $1/\sqrt{D}$ | Simple map + tasklet — Part 1 |
| Softmax (exp, sum, normalize) | Multiple maps, data dependencies — Part 4 |
| $\text{scores} \times V$ matmul | LibraryNode (GEMM) — Part 2 |
| Full analysis pipeline | Arithmetic intensity extractor — Part 4 |
| Targeted transformations | MapTiling on GEMMs — Part 3 |

The complete pipeline:

```
@dace.program → SDFG → expand_library_nodes()
                    ↓
            collect_roofline_data()
                    ↓
     per-component arithmetic intensity table
                    ↓
     targeted transformations on bottlenecks
                    ↓
     quantitative heterogeneity argument
```

---

## 7.2 Writing Self-Attention in DaCe

```python
import dace
import numpy as np

S = dace.symbol('S')   # sequence length
D = dace.symbol('D')   # head dimension

@dace.program
def self_attention(Q: dace.float64[S, D],
                   K: dace.float64[S, D],
                   V: dace.float64[S, D]):
    # Step 1: QK^T — shape [S, S]
    scores = Q @ np.transpose(K)

    # Step 2: Scale
    scores = scores / np.sqrt(D)

    # Step 3: Softmax over each row
    scores_exp = np.exp(scores)
    scores_sum = np.sum(scores_exp, axis=1)
    scores_norm = scores_exp / scores_sum[:, None]

    # Step 4: Weighted sum with V — shape [S, D]
    return scores_norm @ V

sdfg = self_attention.to_sdfg()
sdfg.simplify()
sdfg.save('self_attention.sdfg')
```

---

### Exercise 7.2: Reading the Auto-Generated SDFG

Open `self_attention.sdfg` and answer:

**Q1.** How many states are there and why?

<details>
<summary>Expected answer</summary>

**One state** — because the program has no control flow (no if/else, no
explicit for loops). All four steps of attention are pure dataflow connected
by transient arrays. DaCe correctly places everything in a single state.

Complexity in this SDFG is in the **dataflow graph within the state**, not
across states. This contrasts with the NeSy pattern from Part 6 where
complexity was in the control flow between states.

</details>

---

**Q2.** How many LibraryNodes do you see and what operations do they represent?

<details>
<summary>Expected answer</summary>

Three LibraryNodes before expansion:
- `_Transpose_` — the `np.transpose(K)` call
- `MatMul` (first) — the $QK^T$ multiplication
- `Reduce` — the `np.sum(scores_exp, axis=1)` reduction
- `MatMul` (second) — the $\text{scores\_norm} \times V$ multiplication

After calling `expand_library_nodes()`, each becomes a NestedSDFG with
explicit maps and tasklets inside.

</details>

---

**Q3.** Where does softmax appear in the graph?

<details>
<summary>Expected answer</summary>

Softmax is **not** a single LibraryNode — it is three separate operations
within the same state, connected by transient arrays:

```
scores → _Div__map (÷√D) → __tmp3 → _numpy_exp__map (exp) →
scores_exp → _Div__map_1 (÷sum) → scores_norm
```

DaCe keeps these as separate maps because `scores_exp` feeds two consumers:
the normalize map AND the reduce sum. This data dependency prevents fusion.

</details>

---

## 7.3 Expanding and Inventorying the SDFG

Before running any analysis, always inventory the SDFG structure first.
This prevents surprises in the extractor.

```python
import dace
import re

sdfg = dace.SDFG.from_file('self_attention.sdfg')
sdfg.expand_library_nodes()
sdfg.save('self_attention_expanded.sdfg')

def inventory_sdfg(sdfg, indent=0):
    """
    Print all maps and tasklets with their codes and WCR status.
    Run this before the extractor to understand what patterns exist.
    """
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                print(' ' * indent +
                      f"MAP: {node.label}  range={node.map.range}")
            if isinstance(node, dace.nodes.Tasklet):
                entry = state.entry_node(node)
                wcr = [e.data.wcr for e in state.out_edges(node)
                       if e.data.wcr]
                print(' ' * indent +
                      f"  TASKLET: {node.label}")
                print(' ' * indent +
                      f"    code: '{node.code.as_string.strip()}'")
                print(' ' * indent +
                      f"    map:  {entry.label if entry else 'none'}")
                if wcr:
                    print(' ' * indent + f"    wcr:  {wcr}")
            if isinstance(node, dace.nodes.NestedSDFG):
                print(' ' * indent +
                      f"[NestedSDFG: {node.label}]")
                inventory_sdfg(node.sdfg, indent + 4)

inventory_sdfg(sdfg)
```

**Expected inventory output:**

```
MAP: _Div__map  range=0:S, 0:S
  TASKLET: _Div_    code: '__out = (__in1 / __in2)'
MAP: _numpy_exp__map  range=0:S, 0:S
  TASKLET: _numpy_exp_    code: '__out = exp(__in1)'   ← transcendental
[NestedSDFG: _Transpose_]
    MAP: transpose_map  range=0:S, 0:D
      TASKLET: transpose    code: '__out = __inp'      ← copy, no ops
[NestedSDFG: Reduce]
    MAP: reduce_init_map
      TASKLET: reduce_init  code: '__out = 0'          ← init, skip
    MAP: reduce_output
      TASKLET: identity     code: '__out = __inp'      ← WCR hidden here!
        wcr: ['(lambda x, y: (x + y))']
[NestedSDFG: _MatMult_gemm]
    MAP: gemm_init_map
      TASKLET: gemm_init    code: 'out = 0'            ← init, skip
    MAP: gemm_map
      TASKLET: gemm         code: '__out = (__a * __b)'
        wcr: ['(lambda x, y: (x + y))']
```

> **Key patterns to note before running the extractor:**
> 1. `exp(__in1)` — transcendental, no arithmetic operators → needs function call detection
> 2. `identity` tasklet with WCR — pure assignment but real compute in WCR → check WCR before init filter
> 3. `transpose` and `reduce_init` — genuinely zero FLOPs, should be excluded or shown with 0 FLOPs

---

## 7.4 Running the Arithmetic Intensity Extractor

Use the full extractor from Part 4:

```python
# Import the full extractor from Part 4
# (collect_roofline_data, print_roofline_table, extract_roofline_data)

sdfg = dace.SDFG.from_file('self_attention_expanded.sdfg')

results = extract_roofline_data(
    sdfg,
    symbol_values={'S': 128, 'D': 64}
)
```

**Expected output:**

```
===========================================================================
Component                                         GFLOPs       MB    I (F/B)
===========================================================================
  call_13/_Div__map                             0.000016     0.26     0.0625
  call_13/_numpy_exp__map                       0.000016     0.26     0.0625
  call_13/_Div__map_1                           0.000016     0.26     0.0623
  _Transpose_/_Transpose__state/transpose_map   0.000000     0.13     0.0000
  Reduce/block_0/reduce_output                  0.000016     0.13     0.1240
  _MatMult_gemm/_MatMult_gemm_state/gemm_map    0.002097     0.26     8.0000
  _MatMult_gemm/_MatMult_gemm_state/gemm_map_1  0.002097     0.26     8.0000
===========================================================================
  TOTAL                                         0.004260     1.77     2.4032
===========================================================================
```

---

### Exercise 7.4: Interpreting the Roofline Table

**Q1.** Classify each component as compute-bound or memory-bound.

<details>
<summary>Expected answer</summary>

| Component | I (FLOPs/byte) | Classification | Hardware implication |
|---|---|---|---|
| `_Div__map` (scale ÷√D) | 0.0625 | Strongly memory-bound | Vector unit |
| `_numpy_exp__map` | 0.0625 | Strongly memory-bound | Vector unit |
| `_Div__map_1` (normalize) | 0.0623 | Strongly memory-bound | Vector unit |
| `transpose_map` | 0.0 | Pure data movement | DMA engine |
| `reduce_output` | 0.124 | Memory-bound reduction | Vector unit |
| `gemm_map` ($QK^T$) | 8.000 | Compute-bound | Systolic array |
| `gemm_map_1` ($AV$) | 8.000 | Compute-bound | Systolic array |

A typical CPU ridge point is ~2 FLOPs/byte (100 GFLOPs/s ÷ 50 GB/s). Any
component below the ridge point is memory-bound; above is compute-bound.

</details>

---

**Q2.** What is the ratio between GEMM and pointwise operation intensity?
Why does this matter for hardware design?

<details>
<summary>Expected answer</summary>

$$\frac{I_{\text{GEMM}}}{I_{\text{pointwise}}} = \frac{8.0}{0.0625} = 128\times$$

This 128× difference is the **quantitative heterogeneity argument**:

- A systolic array optimized for GEMMs has high compute density but is
  completely wasted on softmax pointwise ops (memory-bound, needs bandwidth)
- A vector unit optimized for memory bandwidth is massively underutilized
  on GEMMs (compute-bound, needs FLOPs)
- No single accelerator handles both optimally

This single table, automatically derived from the SDFG, is Phase 2 of your
methodology — bottleneck identification and accelerator candidate selection.

</details>

---

**Q3.** Why does the MapFusion transformation fail to fuse the softmax maps?

<details>
<summary>Expected answer</summary>

```python
from dace.transformation.dataflow import MapFusion
sdfg.apply_transformations(MapFusion)  # applies 0 transformations
```

The `scores_exp` array has `out_degree=2` — it feeds both the normalize map
(`_Div__map_1`) AND the reduce sum (`Reduce`). MapFusion's `can_be_applied`
method checks:

```python
if any(graph.out_degree(n) > 1 for n in transients_to_remove):
    return False
```

This is a **real data dependency**, not a tool limitation. Both the normalize
and the reduce need `scores_exp`. You cannot eliminate it without recomputing
`exp` twice or restructuring the algorithm (Flash Attention).

**Research interpretation:** When MapFusion doesn't fire, the SDFG is telling
you something true. The correct optimization here is algorithmic (Flash
Attention), not mechanical (MapFusion).

</details>

---

## 7.5 Applying Targeted Transformations

Based on the intensity table, apply tiling to the two compute-bound GEMMs.

```python
import dace
from dace.transformation.dataflow import MapTiling

sdfg = dace.SDFG.from_file('self_attention_expanded.sdfg')

# Find and tile both gemm_maps
tiled_count = 0
for state in sdfg.states():
    for node in state.nodes():
        if isinstance(node, dace.nodes.NestedSDFG):
            if '_MatMult_gemm' in node.label:
                for nstate in node.sdfg.states():
                    for n in nstate.nodes():
                        if (isinstance(n, dace.nodes.MapEntry)
                                and n.label == 'gemm_map'):
                            print(f"Tiling GEMM: range={n.map.range}")
                            node.sdfg.apply_transformations(
                                MapTiling,
                                options={'tile_sizes': [32, 32, 32]},
                                states=[nstate]
                            )
                            tiled_count += 1

print(f"Tiled {tiled_count} GEMM maps")
sdfg.validate()
sdfg.save('self_attention_tiled.sdfg')
```

---

### Exercise 7.5: Transformations

**Q1.** How many maps does the tiling transformation find and tile?

<details>
<summary>Expected answer</summary>

Two — one for each `_MatMult_gemm` NestedSDFG. The first GEMM is $QK^T$
(range `0:S, 0:S, 0:D`) and the second is $\text{scores} \times V$
(range `0:S, 0:D, 0:S`). Both get tiled with `[32, 32, 32]`.

Note that the ranges differ — the contraction dimension is `D` for $QK^T$
and `S` for the second GEMM. Tiling with 32 in all dimensions handles both.

</details>

---

**Q2.** Does tiling change the arithmetic intensity reported by the extractor?

<details>
<summary>Expected answer</summary>

No — tiling does not change the **static** arithmetic intensity as measured
by the extractor. The boundary memlet subsets (`A[0:M, 0:K]` etc.) remain
the same before and after tiling.

What tiling changes is the **effective** arithmetic intensity seen by the
memory system at runtime — data is reused from cache rather than re-fetched
from DRAM, so actual DRAM traffic is lower. The extractor measures static
transfer volume; the improvement shows up in measured GFLOPs/s from
instrumentation, not in the static intensity calculation.

This is why you need **both** static analysis (extractor) and dynamic
profiling (instrumentation) for a complete roofline picture.

</details>

---

## 7.6 Known Limitations and Future Work

This case study surfaces important limitations worth documenting for your
research:

### Limitation 1: Softmax fusion requires algorithmic restructuring

MapFusion cannot fuse the softmax maps due to the `scores_exp` data
dependency. The extractor correctly identifies three separate memory-bound
passes each at $I = 0.0625$ FLOPs/byte. The correct optimization is
**Flash Attention** — a numerically stable online softmax that computes
exp, sum, and normalize in a single pass. This is an algorithmic change,
not a graph transformation.

**Research implication:** Your methodology correctly identifies this as a
fusion opportunity and flags it as requiring algorithmic restructuring.
This is a genuine research finding — the SDFG analysis reveals *why* the
optimization is blocked.

### Limitation 2: The extractor uses heuristics

The FLOP counter uses regex on tasklet code strings. This works reliably
for:
- Standard arithmetic operators (`+`, `-`, `*`, `/`)
- Transcendental functions (`exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`)
- WCR reductions (`lambda x, y: x + y`)

It may miss:
- Unusual C++ constructs in manual tasklets
- Deeply nested expressions
- Custom intrinsics or SIMD operations

**Mitigation:** Always run `inventory_sdfg()` before the extractor on any
new workload. This surfaces unexpected patterns before they silently produce
wrong counts.

### Limitation 3: Static vs dynamic intensity

The extractor computes **static** arithmetic intensity from array boundary
memlets. This is the theoretical intensity assuming all data comes from DRAM.
The actual intensity at runtime depends on cache behavior.

For a complete picture, combine:
- Static extractor → lower bound on intensity (worst-case DRAM traffic)
- PAPI hardware counters → measured actual DRAM traffic
- The gap between them → cache effectiveness

### Limitation 4: `dace.frontend.torch` availability

The PyTorch frontend (`dace.frontend.torch`) for tracing existing PyTorch
models is not available in DaCe 1.0.2 in the `hypercorex` environment. The
recommended approach for your workloads is to implement key components
directly as `@dace.program` functions — this gives cleaner SDFGs with full
visibility into every kernel.

---

## 7.7 The Complete Methodology Pipeline

You now have every tool needed for Phases 1–3 of your methodology:

```
Phase 1 — Workload Characterization
─────────────────────────────────────────────────────
@dace.program (JEPA / VLM / NeSy component)
    ↓ .to_sdfg() + simplify() + expand_library_nodes()
inventory_sdfg()            ← understand structure first
collect_roofline_data()     ← FLOPs + bytes per component
+ Timer instrumentation     ← wall-clock timing
    ↓
per-component roofline table

Phase 2 — Bottleneck Identification
─────────────────────────────────────────────────────
Classify each component:
    I >> ridge point  → compute-bound → systolic array candidate
    I << ridge point  → memory-bound  → vector unit candidate
    I ≈ 0, bytes > 0  → pure movement → DMA engine candidate
    irregular access  → CPU only      → NeSy symbolic components
    ↓
hardware dispatch map

Phase 3 — Design Space Exploration
─────────────────────────────────────────────────────
For each bottleneck component:
    define Library Node (Part 5)
    implement expansions: pure, cpu_blas, systolic, cva6_mmio
    swap implementations → profile → compare
    ↓
candidate accelerator configurations for gem5-Aladdin (Phase 4)
```

---

### Final Exercise 7.7: Full Pipeline

Apply the complete pipeline to a two-layer transformer encoder. This
combines self-attention with a feed-forward network (FFN):

```python
import dace
import numpy as np

S = dace.symbol('S')   # sequence length
D = dace.symbol('D')   # model dimension
F = dace.symbol('F')   # FFN hidden dimension

@dace.program
def transformer_encoder_layer(
        x:   dace.float64[S, D],
        Wq:  dace.float64[D, D],
        Wk:  dace.float64[D, D],
        Wv:  dace.float64[D, D],
        W1:  dace.float64[D, F],
        W2:  dace.float64[F, D]):

    # ── Self-attention ───────────────────────────────────────────────
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv

    scores     = Q @ np.transpose(K)
    scores     = scores / np.sqrt(D)
    scores_exp = np.exp(scores)
    scores_sum = np.sum(scores_exp, axis=1)
    scores_norm = scores_exp / scores_sum[:, None]
    attn_out   = scores_norm @ V

    # ── Feed-forward network ─────────────────────────────────────────
    h = np.maximum(0.0, attn_out @ W1)   # linear + ReLU
    return h @ W2

sdfg = transformer_encoder_layer.to_sdfg()
sdfg.simplify()
sdfg.expand_library_nodes()
sdfg.save('transformer_encoder.sdfg')
```

**Questions:**

**Q1.** Run `inventory_sdfg()` on the expanded SDFG. How many GEMM maps
do you find and what are their ranges?

<details>
<summary>Expected answer</summary>

Seven GEMMs in total:
- `x @ Wq` — projection: range `[S, D, D]`
- `x @ Wk` — projection: range `[S, D, D]`
- `x @ Wv` — projection: range `[S, D, D]`
- `Q @ K^T` — attention scores: range `[S, S, D]`
- `scores @ V` — attention output: range `[S, D, S]`
- `attn @ W1` — FFN up-projection: range `[S, F, D]`
- `h @ W2` — FFN down-projection: range `[S, D, F]`

The three projection GEMMs ($x \times W_{q,k,v}$) are identical in shape
and should have the same arithmetic intensity. The FFN GEMMs have a
different contraction dimension (F vs D vs S).

</details>

---

**Q2.** Run `extract_roofline_data()` with $S=128$, $D=64$, $F=256$.
Which component has the highest arithmetic intensity? Which has the lowest?

<details>
<summary>Expected answer</summary>

**Highest:** All GEMMs should cluster around $I = 8-32$ FLOPs/byte
depending on their specific shapes. The FFN GEMMs with larger $F=256$
will have higher intensity than the attention GEMMs with $S=128$.

**Lowest:** The softmax pointwise ops ($I = 0.0625$ FLOPs/byte) and
the ReLU activation (also ~0.0625 FLOPs/byte — one op per element, two
array accesses).

The overall heterogeneity ratio should be even larger than the single
self-attention case because the FFN GEMMs with larger dimensions push
intensity higher while softmax remains at the same low intensity.

</details>

---

**Q3.** Based on your roofline table, propose a hardware architecture
for this transformer layer. Which components go to which accelerator?

<details>
<summary>Expected answer</summary>

Proposed heterogeneous architecture:

| Component | Hardware | Reason |
|---|---|---|
| All 7 GEMMs | Systolic array | Compute-bound, regular access |
| Softmax (exp, div, sum) | Vector unit | Memory-bound, element-wise |
| ReLU activation | Vector unit | Memory-bound, element-wise |
| Transpose | DMA engine | Pure data movement, 0 FLOPs |
| Projection weight loads | On-chip SRAM | Reused across sequence positions |

This is exactly the hardware dispatch map that Phase 2 of your methodology
produces. The SDFG analysis fully automated the derivation — you did not
need to read the PyTorch source code to understand the computational
character of each component.

</details>

---

## Summary: Part 7 and Complete Tutorial

You have completed the full DaCe mastery curriculum. Here is what each part
contributed to your research capability:

| Part | Topic | Research capability unlocked |
|---|---|---|
| 1 | SDFG fundamentals | Read any SDFG, understand every node and edge |
| 2 | Matrix multiplication | Understand GEMMs, LibraryNodes, WCR reductions |
| 3 | Transformations | Improve performance, identify optimization limits |
| 4 | Arithmetic intensity | Automate Phase 1+2 of methodology |
| 5 | Library Nodes | Model accelerators, enable Phase 3 DSE |
| 6 | Control flow | Model NeSy dispatch, multi-state workloads |
| 7 | Self-attention | End-to-end methodology on a real workload |

### The core finding

From your self-attention analysis:

$$\frac{I_{\text{GEMM}}}{I_{\text{softmax}}} = \frac{8.0}{0.0625} = 128\times$$

This single number, automatically derived from the SDFG, is the quantitative
foundation of your thesis:

> *Modern AI workloads are fundamentally heterogeneous. The 128× arithmetic
> intensity difference between GEMM and pointwise operations means no single
> compute paradigm can handle them optimally. A principled, data-driven
> methodology — using DaCe as the central backbone — is required to match
> each component to the appropriate hardware.*

### What comes next

| Phase | Tool | Status after this tutorial |
|---|---|---|
| Phase 1: Profiling | DaCe extractor | ✓ Complete |
| Phase 2: Bottleneck ID | Roofline table | ✓ Complete |
| Phase 3: DSE | Library Nodes | ✓ Pattern established |
| Phase 4: Simulation | gem5-Aladdin → SystemC TLM → CVA6 Verilator | Next steps |
| Phase 5: Hardware | RTL → FPGA → ASIC | Final milestone |

The DaCe layer is your foundation. Everything downstream in the toolchain
takes the Library Node implementations and the roofline data you can now
generate automatically from any workload.