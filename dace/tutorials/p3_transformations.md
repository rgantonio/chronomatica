# DaCe Mastery Tutorial
## Part 3: Transformations — MapFusion, MapTiling, and Surgical Targeting

> **Prerequisites:** Complete Parts 1 and 2. You should be comfortable reading
> SDFGs, building them manually, and understanding maps, memlets, and WCR.

---

## 3.1 What Are Transformations?

Transformations are **graph rewriting rules** — they take a subgraph matching
a certain pattern and replace it with an equivalent but differently structured
subgraph. The result computes the same thing but with different performance
characteristics.

This is central to your methodology because:
- **MapFusion** → reduces memory traffic → fewer transient reads/writes
- **MapTiling** → controls cache reuse → directly affects roofline position
- **Schedule changes** → CPU sequential vs multicore vs GPU

### The two most important transformations

| Transformation | What it does | Effect on roofline |
|---|---|---|
| `MapFusion` | Merges two consecutive maps into one | Reduces bytes moved → higher $I$ |
| `MapTiling` | Splits a map into tile + element maps | Reduces DRAM traffic via cache reuse → higher $I$ |

---

## 3.2 MapFusion

MapFusion merges two consecutive maps that are connected through a transient
array. It eliminates the intermediate buffer, reducing memory traffic.

**Before fusion:**
```
Map[i] → AccessNode(__tmp) → Map[i]
```

**After fusion:**
```
Map[i]  (single fused map, no intermediate)
```

### Setup: generate a two-step AXPY

```python
import dace
import numpy as np

N = dace.symbol('N')

@dace.program
def axpy_twostep(a: dace.float64,
                 x: dace.float64[N],
                 y: dace.float64[N]):
    tmp = a * x      # step 1 — produces intermediate
    return tmp + y   # step 2 — consumes intermediate

sdfg = axpy_twostep.to_sdfg()
sdfg.simplify()
sdfg.save('axpy_twostep_before.sdfg')
```

Study the graph before applying any transformation.

---

### Exercise 3.2a: Before Fusion

**Q1.** How many maps are there and what is the transient between them?

<details>
<summary>Expected answer</summary>

There are two maps — one for the multiply (`a * x`) and one for the add
(`tmp + y`). Between them is a transient array `__tmp` (or similar) that
stores the intermediate result. This transient must be written to memory
after the first map and read back before the second.

</details>

---

**Q2.** Count the total memory traffic across both maps. How many array reads
and writes occur?

<details>
<summary>Expected answer</summary>

| Operation | Arrays accessed | Traffic |
|---|---|---|
| Map 1 reads | `a` (scalar), `x[0:N]` | $1 + N$ elements |
| Map 1 writes | `__tmp[0:N]` | $N$ elements |
| Map 2 reads | `__tmp[0:N]`, `y[0:N]` | $2N$ elements |
| Map 2 writes | `__return[0:N]` | $N$ elements |

**Total:** $1 + 5N$ element accesses.

The key insight: the transient `__tmp` costs $2N$ accesses — $N$ writes after
Map 1 and $N$ reads before Map 2.

</details>

---

Now apply MapFusion:

```python
from dace.transformation.dataflow import MapFusion

sdfg = dace.SDFG.from_file('axpy_twostep_before.sdfg')
sdfg.apply_transformations(MapFusion)
sdfg.save('axpy_twostep_after.sdfg')
sdfg.view()
```

---

### Exercise 3.2b: After Fusion

**Q3.** How many maps after fusion? Is the transient still there?

<details>
<summary>Expected answer</summary>

One map — the two maps are merged into a single fused map. The transient
`__tmp` disappears entirely. The fused tasklet computes `_out = _a * _x + _y`
directly, bypassing the intermediate buffer.

</details>

---

**Q4.** What is the new total memory traffic? How much was saved?

<details>
<summary>Expected answer</summary>

| Operation | Arrays accessed | Traffic |
|---|---|---|
| Fused map reads | `a` (scalar), `x[0:N]`, `y[0:N]` | $1 + 2N$ elements |
| Fused map writes | `__return[0:N]` | $N$ elements |

**Total:** $1 + 3N$ element accesses.

**Saved:** $2N$ accesses — the write and read of `__tmp` are eliminated.

**Effect on arithmetic intensity:**
$$I_{\text{before}} = \frac{2N}{(1 + 5N) \times 8} \approx \frac{2}{40} = 0.05 \text{ FLOPs/byte}$$
$$I_{\text{after}} = \frac{2N}{(1 + 3N) \times 8} \approx \frac{2}{24} = 0.083 \text{ FLOPs/byte}$$

Fusion moves the operation rightward on the roofline plot — same FLOPs,
less memory traffic.

</details>

---

### When MapFusion fires — and when it doesn't

MapFusion has strict preconditions in DaCe 1.0.2. It will **not** fire if:

1. The intermediate transient has `out_degree > 1` — it is read by more than
   one consumer. This is a real data dependency, not a tool limitation.
2. The same array name appears multiple times in the state
   (`num_occurrences > 1` check in `can_be_applied`).
3. The two maps have different range structures (incompatible permutations).

> **Research finding:** When MapFusion doesn't fire, the SDFG is telling you
> something true about the computation. For example, in softmax:
>
> ```
> exp_map → scores_exp
>               ├──→ normalize_map
>               └──→ reduce_sum      ← two consumers!
> ```
>
> `scores_exp` feeds both the normalize map and the reduce sum. MapFusion
> correctly refuses — eliminating the transient would break the sum.
> The correct fix is algorithmic restructuring (Flash Attention), not forcing
> fusion.

> **⚠️ Never use `apply_transformations_repeated` for tiling:**
> ```python
> # DANGEROUS — runs forever, hits 100,000 iteration limit
> sdfg.apply_transformations_repeated(MapTiling, ...)
>
> # CORRECT — surgical targeting (see Section 3.3)
> ```
> `apply_transformations_repeated` keeps applying until no matches remain.
> Tiling creates new maps that can themselves be tiled — infinite recursion.

---

## 3.3 MapTiling

MapTiling splits a map into two levels: an outer **tile** map and an inner
**element** map. This controls the working set size for cache reuse.

**Before tiling:**
$$\text{Map}[i: 0:N]$$

**After tiling with tile size T:**
$$\text{Outer tile map}[i: 0:N:T] \times \text{Inner element map}[ii: 0:T]$$

### Why tiling improves arithmetic intensity

Without tiling, the naive matmul accesses rows of $A$ repeatedly — by the time
you come back to reuse a row, it has been evicted from cache. With tiling, you
work on a $T \times T$ block of $C$, reusing the corresponding strips of $A$
and $B$ from cache.

Tiling reduces **DRAM traffic** because data is reused from cache rather than
re-fetched. Total FLOPs stay the same; cache hits replace DRAM misses. So
arithmetic intensity relative to DRAM increases.

### Surgical targeting

Since `apply_transformations_repeated` loops forever, always target specific maps:

```python
import dace
from dace.transformation.dataflow import MapTiling

sdfg = dace.SDFG.from_file('matmul_auto.sdfg')

# Step 1: Find all maps recursively
def find_all_maps(sdfg, indent=0):
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                print(' ' * indent +
                      f"Map: {node.label}  "
                      f"range: {node.map.range}  "
                      f"state: {state.label}")
            if isinstance(node, dace.nodes.NestedSDFG):
                print(' ' * indent + f"[NestedSDFG: {node.label}]")
                find_all_maps(node.sdfg, indent + 4)

find_all_maps(sdfg)
```

Output:
```
[NestedSDFG: _MatMult_gemm]
    Map: gemm_init_map  range: 0:M, 0:N  state: _MatMult_gemm_initstate
    Map: gemm_map  range: 0:M, 0:N, 0:K  state: _MatMult_gemm_state
```

Maps live inside the NestedSDFG — target them directly:

```python
# Step 2: Tile only the gemm_map, not the init map
for state in sdfg.states():
    for node in state.nodes():
        if isinstance(node, dace.nodes.NestedSDFG):
            nested = node.sdfg
            for nested_state in nested.states():
                for n in nested_state.nodes():
                    if (isinstance(n, dace.nodes.MapEntry)
                            and n.label == 'gemm_map'):
                        print(f"Tiling: {n.label}  range: {n.map.range}")
                        nested.apply_transformations(
                            MapTiling,
                            options={'tile_sizes': [32, 32, 32]},
                            states=[nested_state]
                        )

sdfg.validate()
sdfg.save('matmul_tiled.sdfg')
```

---

### Exercise 3.3: MapTiling

**Q1.** After tiling, what does the map structure look like?

<details>
<summary>Expected answer</summary>

The original 3D map `[0:M, 0:N, 0:K]` is split into two levels:

```
Outer tile map:  [0:M:32, 0:N:32, 0:K:32]   ← tile boundaries
Inner tile map:  [tile_i:min(tile_i+31, M-1)+1, ...]  ← within tile
```

The `0:M:32` notation means `start:stop:step` — iterate from 0 to M in steps
of 32. The `Min(...)` expression handles boundary conditions where the array
size is not a perfect multiple of 32.

</details>

---

**Q2.** Why does `apply_transformations` only tile `gemm_init_map` when applied
naively, not `gemm_map`?

<details>
<summary>Expected answer</summary>

`apply_transformations` finds the **first matching subgraph** and applies the
transformation there, then stops. The `gemm_init_map` matched the pattern first
because it appears earlier in the state iteration order.

This is why surgical targeting is essential — you specify exactly which map to
transform rather than relying on pattern matching order.

</details>

---

**Q3.** Should you parallelize at the tile level or the element level? Why?

<details>
<summary>Expected answer</summary>

**Tile-level parallelism is better.** When you parallelize at the tile level,
each thread owns a $32 \times 32$ block of $C$ and all the $A$ and $B$ data
it needs. That data stays in that thread's cache for the duration of the tile
computation — no other thread touches it, so no cache invalidation.

At the element level, threads share rows and columns of $A$ and $B$, constantly
evicting each other's cache lines — defeating the purpose of tiling entirely.

This principle extends to hardware design: each Processing Element (PE) in a
systolic array should own a tile of data, not fight over shared memory.

To set a map's schedule to multicore:

```python
n.map.schedule = dace.ScheduleType.CPU_Multicore
```

</details>

---

## 3.4 Instrumentation and Profiling

DaCe has several instrumentation types for measuring performance:

| Type | Measures | Use for |
|---|---|---|
| `Timer` | Wall-clock latency per node | Identifying slow components |
| `PAPI_Counters` | Hardware events (cache misses, FLOPs, bandwidth) | Measured roofline — real bytes from DRAM |
| `GPU_Events` | CUDA kernel timing | GPU schedule profiling |

### Applying Timer instrumentation

```python
import dace
import numpy as np
import glob
from dace.codegen.instrumentation.report import InstrumentationReport

sdfg = dace.SDFG.from_file('matmul_tiled.sdfg')

# Instrument all maps inside nested SDFGs
for state in sdfg.states():
    for node in state.nodes():
        if isinstance(node, dace.nodes.NestedSDFG):
            for nested_state in node.sdfg.states():
                for n in nested_state.nodes():
                    if isinstance(n, dace.nodes.MapEntry):
                        n.map.instrument = \
                            dace.InstrumentationType.Timer

compiled = sdfg.compile()

A = np.random.rand(512, 256)
B = np.random.rand(256, 512)
C = np.zeros((512, 512))

for _ in range(10):
    compiled(A=A, B=B, C=C, M=512, N=512, K=256)

# Read report
report_files = glob.glob(
    '.dacecache/**/report-*.json', recursive=True)
report = InstrumentationReport(sorted(report_files)[-1])
print(report)
```

### Reading the instrumentation report

The report key structure is:
```
(sdfg_id, state_id, node_id) → {'Map label': {thread_id: [timings...]}}
```

For a parallel map, wall-clock time = **max across all threads** per run
(threads run simultaneously, not sequentially).

Two `gemm_map` entries will appear:
- **Node 0** (inner tile map) — the parallel inner loop, ~0.35ms per thread
- **Node 6** (outer tile map) — the sequential outer loop driving all tiles, ~50ms total

The outer map is sequential by default. Set it to multicore for tile-level
parallelism:

```python
# Find the outer tile map (does not contain 'tile___' in range string)
for n in nested_state.nodes():
    if isinstance(n, dace.nodes.MapEntry):
        if n.label == 'gemm_map':
            if 'tile___' not in str(n.map.range):
                n.map.schedule = dace.ScheduleType.CPU_Multicore
```

---

### Exercise 3.4: Instrumentation

**Q1.** After running the instrumented matmul, compute the achieved GFLOPs/s.

<details>
<summary>Expected answer</summary>

With $M=N=512$, $K=256$:
$$\text{FLOPs} = 2 \times 512 \times 512 \times 256 = 134,217,728 \approx 134 \text{ MFLOPs}$$

Using the wall-clock time from the outer tile map (e.g., ~50ms):
$$\text{Performance} = \frac{134 \text{ MFLOPs}}{0.050 \text{ s}} \approx 2.69 \text{ GFLOPs/s}$$

Typical results on a university server CPU. Compare this to theoretical peak
FLOP/s of your CPU to see how much headroom remains.

</details>

---

**Q2.** Why does the init map (`gemm_init_map`) waste time and how can you
eliminate it?

<details>
<summary>Expected answer</summary>

The init map zeros out $C$ before accumulation. It takes ~2.5ms for a
$512 \times 512$ output — significant overhead for just writing zeros.

Eliminate it by pre-initializing $C$ in NumPy before calling the SDFG:

```python
C = np.zeros((512, 512))   # already zero — init map is redundant
```

This is a common optimization: if the caller guarantees the output buffer is
already zeroed, the init map is pure overhead. Your roofline analysis surfaces
this automatically — the init map shows 0 FLOPs but nonzero bytes, flagging it
as wasteful.

</details>

---

## 3.5 Transformation Workflow for Your Methodology

The correct workflow for applying transformations in your research:

```
Step 1: Generate SDFG and expand library nodes
         ↓
Step 2: Run arithmetic intensity extractor (Part 4)
         ↓
Step 3: Identify bottleneck map by label
         ↓
Step 4: Apply targeted transformation to that specific map
         ↓
Step 5: Re-run extractor — measure intensity improvement
         ↓
Step 6: Instrument and profile — measure performance improvement
```

Never apply transformations blindly to the whole SDFG. Always profile first,
identify the bottleneck, then target surgically.

---

## Summary: Part 3

You now understand:

- **MapFusion** eliminates transient arrays between consecutive maps,
  reducing memory traffic and improving arithmetic intensity
- **MapFusion preconditions**: fires only when the intermediate transient
  has a single consumer — when it doesn't fire, there's a real data dependency
- **MapTiling** splits maps into tile + element levels for cache blocking —
  use surgical targeting, never `apply_transformations_repeated`
- **Tile-level parallelism** is better than element-level for cache coherency
- **Instrumentation** gives you wall-clock timing per map — combine with
  FLOPs from the extractor (Part 4) to get GFLOPs/s
- The outer tile map is **sequential by default** — set it to
  `CPU_Multicore` for full parallelism

**Key API fix to remember:**
> `apply_transformations_repeated(MapTiling)` loops forever — always use
> surgical targeting by walking the graph and calling
> `apply_transformations(MapTiling, states=[specific_state])` on the
> exact map you want to tile.

**Next:** Part 4 covers the arithmetic intensity extractor — building a
symbolic FLOP and bytes counter that runs automatically on any SDFG and
produces roofline data for every component.