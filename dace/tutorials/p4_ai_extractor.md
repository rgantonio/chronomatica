# DaCe Mastery Tutorial
## Part 4: Arithmetic Intensity Extractor

> **Prerequisites:** Complete Parts 1–3. You should understand maps, memlets,
> WCR, NestedSDFGs, and the roofline model concept.

---

## 4.1 Why Build an Extractor?

The roofline model places each component at a point:

$$\text{Roofline point} = (I, P) = \left(\frac{\text{FLOPs}}{\text{Bytes}},\ \frac{\text{FLOPs}}{t_{\text{wall}}}\right)$$

DaCe does not provide a built-in roofline extractor. It provides:
- **Instrumentation** — timing (what we used in Part 3)
- **Graph structure** — map ranges, memlet subsets, tasklet code

Your extractor builds the bridge: walk the SDFG graph, extract FLOPs and bytes
symbolically using SymPy, substitute concrete values, and combine with timing.

This is not something DaCe provides out of the box.
---

## 4.2 What Data Is Extractable from Maps

Before writing the extractor, understand what raw data the SDFG exposes.

```python
import dace
import sympy

sdfg = dace.SDFG.from_file('matmul_auto.sdfg')

def inspect_maps(sdfg, indent=0):
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                print(' ' * indent + f"Map: {node.label}")
                print(' ' * indent + f"  ranges: {node.map.range}")
                for i, r in enumerate(node.map.range):
                    start, stop, step = r
                    # Correct iteration count formula
                    count = (stop - start + 1) / step
                    print(' ' * indent +
                          f"  dim {i}: start={start} stop={stop} "
                          f"step={step}  iterations={count}")
            if isinstance(node, dace.nodes.Tasklet):
                print(' ' * indent + f"Tasklet: {node.label}")
                print(' ' * indent +
                      f"  code: {node.code.as_string}")
            if isinstance(node, dace.nodes.NestedSDFG):
                print(' ' * indent + f"[NestedSDFG: {node.label}]")
                inspect_maps(node.sdfg, indent + 4)

inspect_maps(sdfg)
```

> **Important: the off-by-one fix**
>
> DaCe stores range as `(start, stop, step)` where `stop = N - 1` for a range
> `0:N`. So the correct iteration count formula is:
> $$\text{iterations} = \frac{\text{stop} - \text{start}}{\text{step}} + 1 = \frac{(N-1) - 0}{1} + 1 = N$$
>
> Without the `+ 1`, you get $N - 1$ which is wrong.

---

### Exercise 4.2: Inspecting Map Data

**Q1.** What Python type are the map range values — concrete numbers or
something else?

<details>
<summary>Expected answer</summary>

They are **SymPy symbolic expressions**. Since `M`, `N`, `K` are declared as
`dace.symbol(...)`, DaCe keeps them symbolic throughout. The range components
are SymPy objects that support symbolic arithmetic:

```python
start, stop, step = node.map.range[0]
count = (stop - start + 1) / step
print(count)        # M  (SymPy expression)
print(count ** 2)   # M**2
# Substitute concrete values:
concrete = count.subs({'M': 512})
print(concrete)     # 512
```

This is exactly what you need for roofline analysis — compute symbolic FLOP
expressions like $2MNK$, then substitute concrete sizes to get numbers.

</details>

---

**Q2.** What does the tasklet code string look like, and what does `void` mean
on the connector types?

<details>
<summary>Expected answer</summary>

The tasklet code is a plain Python string: `'__out = (__a * __b)'`. This string
is emitted **verbatim** into the generated C++ file — DaCe does no analysis of
it beyond pattern matching for FLOP counting.

The `void` connector type does **not** mean the data type is void in the
generated code. It means DaCe hasn't assigned a type annotation to that
connector. The actual data type is inferred from the connected memlets and
array declarations during C++ code generation. `__a: void` in Python becomes
`double __a` in the generated C++.

</details>

---

## 4.3 Building the FLOP Counter

The FLOP count per map is:
$$\text{FLOPs} = \text{iterations}_{dim_0} \times \text{iterations}_{dim_1} \times \cdots \times \text{ops\_per\_iteration}$$

The tricky part is counting ops from the tasklet code string. We use regex
heuristics for arithmetic operators and function calls.

### Key fixes required

**Fix 1 — Init tasklets must be excluded**

Tasklets like `out = 0` or `__out = __inp` (pure assignment) are not compute
operations. Counting them inflates FLOPs incorrectly.

Detection: no arithmetic operators AND no function calls = init tasklet.

```python
def is_init_tasklet(code_str):
    code = code_str.strip()
    has_arithmetic    = bool(re.search(r'[\+\-\*\/]', code))
    has_function_call = bool(re.search(r'\w+\s*\(', code))
    return not has_arithmetic and not has_function_call
```

**Fix 2 — WCR ops are hidden in memlet edges, not tasklet code**

The accumulation `+ y` in a reduction is in the WCR lambda, not the tasklet:

```python
# Tasklet only has multiply:
code = '__out = (__a * __b)'    # 1 op counted

# The add is here — in the output memlet WCR:
wcr = 'lambda x, y: x + y'     # 1 op — missed without this fix!
```

Always check `edge.data.wcr` on tasklet output edges.

**Fix 3 — Transcendental functions count as 1 op each**

`exp(__in1)` has no arithmetic operators — regex misses it entirely. Add
explicit detection for transcendental function calls.

**Fix 4 — WCR-only tasklets must not be filtered**

A tasklet with code `__out = __inp` (pure assignment) looks like an init tasklet,
but if it has a WCR on its output edge it is performing real computation.
Check WCR *before* applying the init filter.

---

### Full FLOP counter implementation

```python
import dace
import sympy
import re

def is_init_tasklet(code_str):
    """Detect pure assignment/init tasklets with no arithmetic."""
    code = code_str.strip()
    has_arithmetic    = bool(re.search(r'[\+\-\*\/]', code))
    has_function_call = bool(re.search(r'\w+\s*\(', code))
    return not has_arithmetic and not has_function_call

def count_tasklet_ops(tasklet, state):
    """
    Count ops in tasklet code AND outgoing WCR memlets.
    WCR ops are checked first — even assignment-looking tasklets
    can have real compute hidden in a WCR.
    """
    code = tasklet.code.as_string.strip()

    # Step 1: check WCR on output edges first
    wcr_ops = 0
    for edge in state.out_edges(tasklet):
        if edge.data.wcr is not None:
            wcr_str = edge.data.wcr
            wcr_ops += len(re.findall(
                r'(?<![<>!])[\+\-](?!=)', wcr_str))
            wcr_ops += len(re.findall(r'\*(?!=)', wcr_str))

    # Step 2: if pure assignment but has WCR — count WCR ops only
    if is_init_tasklet(code):
        return wcr_ops   # 0 if no WCR, >0 if reduction

    # Step 3: count tasklet body ops
    multiplies      = len(re.findall(r'\*(?!=)', code))
    adds            = len(re.findall(r'(?<![<>!])[\+\-](?!=)', code))
    divides         = len(re.findall(r'/(?!=)', code))
    transcendentals = len(re.findall(
        r'\b(exp|log|sqrt|sin|cos|tan|tanh|sigmoid)\s*\(', code))

    tasklet_ops = multiplies + adds + divides + transcendentals
    return max(tasklet_ops + wcr_ops, 1)

def get_map_iterations(map_entry):
    """Compute total iteration count across all map dimensions."""
    total = sympy.Integer(1)
    for r in map_entry.map.range:
        start, stop, step = r
        # Correct formula: (stop - start) / step + 1
        total = total * sympy.simplify((stop - start + 1) / step)
    return sympy.simplify(total)

def get_flops_for_map(node, state):
    """Count FLOPs for a map entry node."""
    tasklet_ops = 0
    for inner_node in state.nodes():
        if isinstance(inner_node, dace.nodes.Tasklet):
            if state.entry_node(inner_node) == node:
                tasklet_ops += count_tasklet_ops(inner_node, state)
    if tasklet_ops == 0:
        return sympy.Integer(0)
    return sympy.simplify(get_map_iterations(node) * tasklet_ops)
```

---

### Exercise 4.3: FLOP Counter

Run the FLOP counter on `matmul_auto.sdfg` (expanded) and answer:

**Q1.** What symbolic FLOP expression does it produce for `gemm_map`?

<details>
<summary>Expected answer</summary>

`K*M*N` — the product of all three map dimensions times 1 op/iteration from
the tasklet. But wait — this is only half right. The multiply gives 1 op, but
the WCR adds another 1. So with the WCR fix applied you should see `2*K*M*N`.

If you only get `K*M*N`, verify that `count_tasklet_ops` is checking
`edge.data.wcr` on the tasklet's output edges.

</details>

---

**Q2.** Does `gemm_init_map` appear in the output? Should it?

<details>
<summary>Expected answer</summary>

It should **not** appear. The `gemm_init` tasklet has code `out = 0` — a pure
assignment with no arithmetic and no function call. `is_init_tasklet` correctly
returns `True` and `count_tasklet_ops` returns 0. The map is then filtered out
since FLOPs == 0.

This is correct behavior — initializing to zero is not a compute operation.
Counting it would inflate FLOPs and make arithmetic intensity look artificially
higher.

</details>

---

**Q3.** With $M=N=512$, $K=256$, what is the concrete FLOP count?

<details>
<summary>Expected answer</summary>

$$2 \times M \times N \times K = 2 \times 512 \times 512 \times 256 = 134,217,728 \text{ FLOPs} \approx 134 \text{ MFLOPs}$$

</details>

---

## 4.4 Building the Bytes Counter

The bytes moved comes from memlet subsets on edges **crossing map boundaries**:
- Edges entering a map scope (AccessNode → MapEntry): data coming **in**
- Edges leaving a map scope (MapExit → AccessNode): data going **out**

Only boundary edges are counted — not internal edges inside the map scope.

$$\text{Bytes} = \sum_{\text{boundary edges}} \text{volume} \times \text{element\_size}$$

### Critical fix: use `current_sdfg` for array lookups

When recursing into nested SDFGs, array lookups must use the **nested SDFG's**
array dictionary, not the parent's. Otherwise `m.data in sdfg.arrays` returns
`False` and bytes are silently counted as zero.

```python
def get_subset_volume(subset):
    """Compute number of elements in a memlet subset."""
    volume = sympy.Integer(1)
    for dim in subset:
        start, stop, step = dim
        volume = volume * sympy.simplify((stop - start) / step + 1)
    return sympy.simplify(volume)

def get_bytes_for_map(node, state, current_sdfg):
    """
    Count bytes crossing map boundary edges.
    current_sdfg: the SDFG that owns this state's arrays.
    Must pass the correct nested SDFG when recursing.
    """
    total_bytes = sympy.Integer(0)
    map_exit = state.exit_node(node)

    # Incoming edges (reads)
    for edge in state.in_edges(node):
        m = edge.data
        if m.data is not None and m.subset is not None:
            if m.data in current_sdfg.arrays:   # ← use current_sdfg
                arr = current_sdfg.arrays[m.data]
                total_bytes += (get_subset_volume(m.subset)
                                * arr.dtype.bytes)

    # Outgoing edges from map exit (writes)
    for edge in state.out_edges(map_exit):
        m = edge.data
        if m.data is not None and m.subset is not None:
            if m.data in current_sdfg.arrays:   # ← use current_sdfg
                arr = current_sdfg.arrays[m.data]
                total_bytes += (get_subset_volume(m.subset)
                                * arr.dtype.bytes)

    return sympy.simplify(total_bytes)
```

---

### Exercise 4.4: Bytes Counter

Run the bytes counter on `matmul_auto.sdfg` (expanded) and answer:

**Q1.** What symbolic byte expressions appear for `gemm_map`?

<details>
<summary>Expected answer</summary>

Three boundary edges, all float64 (8 bytes each):

| Edge | Array | Subset | Bytes |
|---|---|---|---|
| IN `_a` | A matrix | `0:M, 0:K` | $8 \times M \times K$ |
| IN `_b` | B matrix | `0:K, 0:N` | $8 \times K \times N$ |
| OUT `_c` | C matrix | `0:M, 0:N` | $8 \times M \times N$ |

**Total:** $8(MK + KN + MN)$

Note: the volume on `_a` is $M \times K$, not $M \times K \times N$ — the
boundary memlet reflects the **array size**, not the total access count across
all iterations. The $N$-fold reuse of A is handled by the roofline model, not
the memlet volume.

</details>

---

**Q2.** With $M=N=512$, $K=256$, what is the total byte count?

<details>
<summary>Expected answer</summary>

$$8(MK + KN + MN) = 8(512 \times 256 + 256 \times 512 + 512 \times 512)$$
$$= 8(131072 + 131072 + 262144) = 8 \times 524288 = 4,194,304 \text{ bytes} \approx 4.19 \text{ MB}$$

</details>

---

## 4.5 The Combined Roofline Extractor

Combine FLOP counting, bytes counting, and timing into a single pipeline.

### Architecture: separate collection from printing

A key design decision: **collect data recursively, print only once at the top
level**. If you print inside the recursive function, you get multiple partial
tables — one for every nested SDFG level.

```
collect_roofline_data(sdfg)   ← pure recursion, no printing
         ↓
     flat dict of results
         ↓
print_roofline_table(results) ← called once at top level
```

### Deduplication: handle identical labels

Two nested SDFGs with the same label (e.g., two `_MatMult_gemm` nodes) will
produce duplicate keys and overwrite each other. Use a shared counter:

```python
# In collect_roofline_data:
base_label = f"{prefix}{state.label}/{node.label}"
count = counters.get(base_label, 0)
counters[base_label] = count + 1
label = f"{base_label}_{count}" if count > 0 else base_label
```

### Symbolic comparison fix

Never use `>` to compare a SymPy expression against a number:

```python
# Wrong — raises TypeError for symbolic expressions
if bytes_moved > 0:
    intensity = flops / bytes_moved

# Correct — structural equality check works symbolically
intensity = (sympy.simplify(flops / bytes_moved)
             if bytes_moved != sympy.Integer(0)
             else sympy.oo)
```

### Full implementation

```python
import dace
import sympy
import re
import numpy as np
import glob
from dace.codegen.instrumentation.report import InstrumentationReport

# ── FLOP counter ────────────────────────────────────────────────────

def is_init_tasklet(code_str):
    code = code_str.strip()
    has_arithmetic    = bool(re.search(r'[\+\-\*\/]', code))
    has_function_call = bool(re.search(r'\w+\s*\(', code))
    return not has_arithmetic and not has_function_call

def count_tasklet_ops(tasklet, state):
    code = tasklet.code.as_string.strip()
    wcr_ops = 0
    for edge in state.out_edges(tasklet):
        if edge.data.wcr is not None:
            wcr_str = edge.data.wcr
            wcr_ops += len(re.findall(
                r'(?<![<>!])[\+\-](?!=)', wcr_str))
            wcr_ops += len(re.findall(r'\*(?!=)', wcr_str))
    if is_init_tasklet(code):
        return wcr_ops
    multiplies      = len(re.findall(r'\*(?!=)', code))
    adds            = len(re.findall(r'(?<![<>!])[\+\-](?!=)', code))
    divides         = len(re.findall(r'/(?!=)', code))
    transcendentals = len(re.findall(
        r'\b(exp|log|sqrt|sin|cos|tan|tanh|sigmoid)\s*\(', code))
    tasklet_ops = multiplies + adds + divides + transcendentals
    return max(tasklet_ops + wcr_ops, 1)

def get_map_iterations(map_entry):
    total = sympy.Integer(1)
    for r in map_entry.map.range:
        start, stop, step = r
        total = total * sympy.simplify((stop - start + 1) / step)
    return sympy.simplify(total)

def get_flops_for_map(node, state):
    tasklet_ops = 0
    for inner_node in state.nodes():
        if isinstance(inner_node, dace.nodes.Tasklet):
            if state.entry_node(inner_node) == node:
                tasklet_ops += count_tasklet_ops(inner_node, state)
    if tasklet_ops == 0:
        return sympy.Integer(0)
    return sympy.simplify(get_map_iterations(node) * tasklet_ops)

# ── Bytes counter ────────────────────────────────────────────────────

def get_subset_volume(subset):
    volume = sympy.Integer(1)
    for dim in subset:
        start, stop, step = dim
        volume = volume * sympy.simplify((stop - start) / step + 1)
    return sympy.simplify(volume)

def get_bytes_for_map(node, state, current_sdfg):
    total_bytes = sympy.Integer(0)
    map_exit = state.exit_node(node)
    for edge in state.in_edges(node):
        m = edge.data
        if m.data is not None and m.subset is not None:
            if m.data in current_sdfg.arrays:
                arr = current_sdfg.arrays[m.data]
                total_bytes += (get_subset_volume(m.subset)
                                * arr.dtype.bytes)
    for edge in state.out_edges(map_exit):
        m = edge.data
        if m.data is not None and m.subset is not None:
            if m.data in current_sdfg.arrays:
                arr = current_sdfg.arrays[m.data]
                total_bytes += (get_subset_volume(m.subset)
                                * arr.dtype.bytes)
    return sympy.simplify(total_bytes)

# ── Collector ────────────────────────────────────────────────────────

def collect_roofline_data(sdfg, prefix='', counters=None):
    """
    Recursively collect roofline data. No printing — pure collection.
    Returns flat dict: {label: {flops, bytes, intensity}}
    counters: shared dict for deduplicating identical labels.
    """
    if counters is None:
        counters = {}

    results = {}

    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                flops       = get_flops_for_map(node, state)
                bytes_moved = get_bytes_for_map(node, state, sdfg)

                # Skip zero-FLOP zero-byte nodes (pure init maps)
                if (flops == sympy.Integer(0)
                        and bytes_moved == sympy.Integer(0)):
                    continue

                # Symbolic comparison fix — never use > with SymPy
                intensity = (sympy.simplify(flops / bytes_moved)
                             if bytes_moved != sympy.Integer(0)
                             else sympy.oo)

                # Deduplicate labels
                base_label = f"{prefix}{state.label}/{node.label}"
                count = counters.get(base_label, 0)
                counters[base_label] = count + 1
                label = (f"{base_label}_{count}"
                         if count > 0 else base_label)

                results[label] = {
                    'flops':     flops,
                    'bytes':     bytes_moved,
                    'intensity': intensity
                }

            if isinstance(node, dace.nodes.NestedSDFG):
                nested = collect_roofline_data(
                    node.sdfg,
                    prefix=f"{prefix}{node.label}/",
                    counters=counters
                )
                results.update(nested)

    return results

# ── Printer ──────────────────────────────────────────────────────────

def print_roofline_table(results, symbol_values):
    """Print formatted roofline table. Call once at top level only."""
    print(f"\n{'='*75}")
    print(f"{'Component':<45} {'GFLOPs':>10} {'MB':>8} {'I (F/B)':>10}")
    print(f"{'='*75}")

    total_flops = 0
    total_bytes = 0

    for label, data in results.items():
        try:
            f = int(data['flops'].subs(symbol_values))
            b = int(data['bytes'].subs(symbol_values))
            i = f / b if b > 0 else float('inf')
            total_flops += f
            total_bytes += b
            print(f"  {label:<43} "
                  f"{f/1e9:>10.6f} "
                  f"{b/1e6:>8.2f} "
                  f"{i:>10.4f}")
        except Exception as e:
            print(f"  {label}: (symbolic) {e}")

    print(f"{'='*75}")
    if total_bytes > 0:
        print(f"  {'TOTAL':<43} "
              f"{total_flops/1e9:>10.6f} "
              f"{total_bytes/1e6:>8.2f} "
              f"{total_flops/total_bytes:>10.4f}")
    else:
        print(f"  {'TOTAL':<43} "
              f"{total_flops/1e9:>10.6f} "
              f"{total_bytes/1e6:>8.2f} "
              f"{'N/A':>10}")
    print(f"{'='*75}")

# ── Report reader ─────────────────────────────────────────────────────

def read_report(report_path):
    """Parse instrumentation report into {map_label: timing_dict}."""
    report = InstrumentationReport(report_path)
    timings = {}
    for key, entry in report.durations.items():
        for label, thread_data in entry.items():
            clean_label = label.replace('Map ', '').strip()
            all_thread_lists = list(thread_data.values())
            n_runs = len(all_thread_lists[0])
            wall_times = []
            for run_idx in range(n_runs):
                run_max = max(
                    tl[run_idx]
                    for tl in all_thread_lists
                    if run_idx < len(tl)
                )
                wall_times.append(run_max)
            timings[clean_label] = {
                'mean_ms':   np.mean(wall_times),
                'min_ms':    np.min(wall_times),
                'median_ms': np.median(wall_times),
                'node_id':   key[2]
            }
    return timings

# ── Full analysis ─────────────────────────────────────────────────────

def extract_roofline_data(sdfg, symbol_values=None):
    """Collect and optionally print roofline data."""
    results = collect_roofline_data(sdfg)
    if symbol_values:
        print_roofline_table(results, symbol_values)
    return results

def full_roofline_analysis(sdfg_path, symbol_values,
                           report_path=None):
    """
    Complete roofline analysis combining intensity + timing.
    """
    sdfg = dace.SDFG.from_file(sdfg_path)
    results = collect_roofline_data(sdfg)
    timing_data = read_report(report_path) if report_path else {}

    has_timing = bool(report_path)
    width = 90 if has_timing else 75
    header = (f"{'Component':<45} {'GFLOPs':>10} {'MB':>8} "
              f"{'I (F/B)':>8} {'ms':>8} {'GF/s':>8}"
              if has_timing else
              f"{'Component':<45} {'GFLOPs':>10} {'MB':>8} {'I (F/B)':>10}")

    print(f"\n{'='*width}")
    print(header)
    print(f"{'='*width}")

    total_flops = 0
    total_bytes = 0

    for label, data in results.items():
        try:
            f = int(data['flops'].subs(symbol_values))
            b = int(data['bytes'].subs(symbol_values))
            i = f / b if b > 0 else float('inf')
            total_flops += f
            total_bytes += b

            if has_timing:
                short_label = re.sub(r'_\d+$', '',
                                     label.split('/')[-1])
                timing = None
                for tl, td in timing_data.items():
                    if short_label in tl:
                        if (timing is None or
                                td['node_id'] < timing['node_id']):
                            timing = td
                if timing:
                    t_ms = timing['mean_ms']
                    gfs  = f / 1e9 / (t_ms / 1000.0)
                    print(f"  {label:<43} {f/1e9:>10.6f} "
                          f"{b/1e6:>8.2f} {i:>8.4f} "
                          f"{t_ms:>8.3f} {gfs:>8.3f}")
                else:
                    print(f"  {label:<43} {f/1e9:>10.6f} "
                          f"{b/1e6:>8.2f} {i:>8.4f} "
                          f"{'N/A':>8} {'N/A':>8}")
            else:
                print(f"  {label:<43} {f/1e9:>10.6f} "
                      f"{b/1e6:>8.2f} {i:>10.4f}")

        except Exception as e:
            print(f"  {label}: (symbolic) {e}")

    print(f"{'='*width}")
    if total_bytes > 0:
        print(f"  {'TOTAL':<43} "
              f"{total_flops/1e9:>10.6f} "
              f"{total_bytes/1e6:>8.2f} "
              f"{total_flops/total_bytes:>10.4f}")
    else:
        print(f"  {'TOTAL':<43} "
              f"{total_flops/1e9:>10.6f} "
              f"{total_bytes/1e6:>8.2f} "
              f"{'N/A':>10}")
    print(f"{'='*width}")
    return results


# ── Test on matmul ────────────────────────────────────────────────────

if __name__ == '__main__':
    import dace
    import numpy as np

    M, N, K = dace.symbol('M'), dace.symbol('N'), dace.symbol('K')

    @dace.program
    def matmul(A: dace.float64[M, K], B: dace.float64[K, N]):
        return A @ B

    sdfg = matmul.to_sdfg()
    sdfg.simplify()
    sdfg.expand_library_nodes()
    sdfg.save('matmul_auto.sdfg')

    symbol_values = {'M': 512, 'N': 512, 'K': 256}

    print("=== Intensity only ===")
    extract_roofline_data(sdfg, symbol_values=symbol_values)
```

Expected output:
```
===========================================================================
Component                                         GFLOPs       MB    I (F/B)
===========================================================================
  _MatMult_gemm/_MatMult_gemm_state/gemm_map    0.134218     4.19    32.0000
===========================================================================
  TOTAL                                         0.134218     4.19    32.0000
===========================================================================
```

---

### Exercise 4.5: Full Extractor

**Q1.** The arithmetic intensity for matmul is exactly 32.0 FLOPs/byte. Verify
this manually from first principles.

<details>
<summary>Expected answer</summary>

$$I = \frac{\text{FLOPs}}{\text{Bytes}} = \frac{2MNK}{8(MK + KN + MN)}$$

With $M=N=512$, $K=256$:
$$I = \frac{2 \times 512 \times 512 \times 256}{8(512 \times 256 + 256 \times 512 + 512 \times 512)}$$
$$= \frac{134,217,728}{8 \times 524,288} = \frac{134,217,728}{4,194,304} = 32.0 \text{ FLOPs/byte} ✓$$

</details>

---

**Q2.** Run the extractor on your self-attention SDFG. What does the intensity
difference between GEMMs and pointwise ops tell you about hardware requirements?

<details>
<summary>Expected answer</summary>

Expected output for self-attention with $S=128$, $D=64$:

| Component | I (FLOPs/byte) | Character |
|---|---|---|
| `_Div__map` (scale) | 0.0625 | Strongly memory-bound |
| `_numpy_exp__map` | 0.0625 | Strongly memory-bound |
| `_Div__map_1` (normalize) | 0.0623 | Strongly memory-bound |
| `reduce_output` | 0.124 | Memory-bound reduction |
| `gemm_map` (QKT) | 8.000 | Compute-bound |
| `gemm_map_1` (AV) | 8.000 | Compute-bound |

$$\frac{I_{\text{GEMM}}}{I_{\text{pointwise}}} = \frac{8.0}{0.0625} = 128\times$$

The GEMMs are 128× more compute-intensive than the pointwise operations.
This is the **quantitative heterogeneity argument**:
- A systolic array optimized for GEMMs is wasted on softmax pointwise ops
- A vector unit optimized for memory bandwidth is underutilized on GEMMs
- No single accelerator handles both optimally

This single table is Phase 2 of your methodology — bottleneck identification
and accelerator candidate selection — fully automated from the SDFG.

</details>

---

**Q3.** Why does the extractor need to be rebuilt for new workloads sometimes?
What is the robust solution?

<details>
<summary>Expected answer</summary>

The extractor uses heuristics (regex on tasklet code strings) that can miss
unusual patterns:
- Transcendental functions like `exp()` have no arithmetic operators
- WCR-only tasklets look like init tasklets without the WCR check
- Deeply nested SDFGs may have arrays in a different scope

The robust solution is to run an **SDFG inspector first** before the extractor:

```python
def inspect_sdfg_structure(sdfg):
    """Print all tasklets and their WCR status before counting."""
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet):
                wcr = [e.data.wcr for e in state.out_edges(node)
                       if e.data.wcr]
                print(f"Tasklet: {node.label}")
                print(f"  code: {node.code.as_string}")
                print(f"  wcr:  {wcr}")
            if isinstance(node, dace.nodes.NestedSDFG):
                inspect_sdfg_structure(node.sdfg)
```

Running this before the extractor tells you exactly what patterns exist,
so you know what to expect and can patch any missed cases.

</details>

---

## 4.6 Combining with Timing Data

Once you have an instrumentation report (from Part 3), combine it with
intensity data:

```python
# Generate and instrument the SDFG
sdfg = dace.SDFG.from_file('matmul_auto.sdfg')

for state in sdfg.states():
    for node in state.nodes():
        if isinstance(node, dace.nodes.NestedSDFG):
            for ns in node.sdfg.states():
                for n in ns.nodes():
                    if isinstance(n, dace.nodes.MapEntry):
                        n.map.instrument = \
                            dace.InstrumentationType.Timer

compiled = sdfg.compile()
A = np.random.rand(512, 256)
B = np.random.rand(256, 512)
C = np.zeros((512, 512))
for _ in range(10):
    compiled(A=A, B=B, C=C, M=512, N=512, K=256)

# Run full analysis
report_files = sorted(glob.glob(
    '.dacecache/**/report-*.json', recursive=True))

full_roofline_analysis(
    sdfg_path='matmul_auto.sdfg',
    symbol_values={'M': 512, 'N': 512, 'K': 256},
    report_path=report_files[-1]
)
```

Expected output:
```
==========================================================================================
Component                                   GFLOPs       MB   I (F/B)       ms     GF/s
==========================================================================================
  _MatMult_gemm/.../gemm_map              0.134218     4.19    32.00    0.370    0.362
==========================================================================================
```

The `GF/s` column combined with `I (F/B)` gives you your complete roofline
point — one dot on the roofline plot with both coordinates fully determined
from the SDFG.

---

## Summary: Part 4

You now have a complete arithmetic intensity extraction pipeline:

```
SDFG
  ↓ collect_roofline_data()
  ├── get_flops_for_map()
  │     ├── get_map_iterations()  [SymPy symbolic]
  │     └── count_tasklet_ops()   [regex + WCR check]
  └── get_bytes_for_map()
        └── get_subset_volume()   [boundary edges only]
  ↓
flat dict {label: {flops, bytes, intensity}}
  ↓ print_roofline_table() / full_roofline_analysis()
  ↓
per-component roofline table
```

**Key fixes to remember:**

| Fix | Why it matters |
|---|---|
| `(stop - start + 1) / step` | Off-by-one in iteration count |
| Check WCR before init filter | Reductions look like assignments |
| Transcendental regex | `exp()` has no arithmetic operators |
| Pass `current_sdfg` | Nested SDFG array lookup must use local scope |
| Symbolic comparison `!= sympy.Integer(0)` | `> 0` raises TypeError for symbolic |
| Separate collect from print | Recursive printing gives multiple partial tables |
| Deduplicate labels with counter | Two GEMMs with same name overwrite each other |

**Next:** Part 5 covers Library Nodes — defining custom accelerator nodes with
multiple implementations for design space exploration.