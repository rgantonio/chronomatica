# DaCe Mastery Tutorial
## Part 5: Library Nodes and Accelerator Modeling

> **Prerequisites:** Complete Parts 1–4. You should understand maps, memlets,
> WCR, NestedSDFGs, and the arithmetic intensity extractor.

---

## 5.1 What Is a Library Node?

A **Library Node** is DaCe's mechanism for representing an **abstract operation
with multiple possible implementations**. It appears as a hexagon in the SDFG
viewer and can be swapped between implementations without changing the
surrounding graph structure.

This is the core mechanism for your accelerator design space exploration:

```
LibraryNode (abstract: "compute MAC")
     ├── pure        → explicit maps + tasklets (correctness reference)
     ├── cpu_blas    → calls optimized BLAS library
     ├── systolic    → your systolic array accelerator model
     └── cva6_mmio   → MMIO calls to physical CVA6 hardware
```

One SDFG, four hardware targets, zero code duplication.

---

## 5.2 The Three-Part Pattern

Every Library Node requires three components:

```
1. Expansion class(es)   → concrete implementations
2. Node class            → abstract interface + implementation registry
3. Usage                 → instantiate, connect, expand, compile
```

This three-part routine is your standard pattern for every accelerator kernel
you model.

---

## 5.3 Building a MAC Library Node

We'll build a Multiply-Accumulate (MAC) node — small enough to understand the
full pattern, directly relevant to your accelerator work.

The operation: `result = sum(x * y)` — a dot product.

### Part A: The Pure Expansion

```python
import dace
from dace.transformation.transformation import ExpandTransformation

@dace.library.expansion
class ExpandMACPure(ExpandTransformation):
    """
    Pure CPU expansion — explicit maps and tasklets.
    Used for correctness validation and profiling.
    """
    environments = []   # no external dependencies

    @staticmethod
    def expansion(node, state, sdfg):
        """
        Called by expand_library_nodes().
        Receives the abstract node and returns a concrete nested SDFG.
        node.n gives the problem size declared on the Library Node.
        """
        N = node.n

        nsdfg = dace.SDFG('mac_pure')
        nsdfg.add_array('_x', shape=[N], dtype=dace.float64)
        nsdfg.add_array('_y', shape=[N], dtype=dace.float64)
        nsdfg.add_array('_result', shape=[1], dtype=dace.float64)

        nstate = nsdfg.add_state('mac_compute')

        x_node = nstate.add_read('_x')
        y_node = nstate.add_read('_y')
        r_node = nstate.add_write('_result')

        me, mx = nstate.add_map('mac_map', {'i': f'0:{N}'})

        tasklet = nstate.add_tasklet(
            'mac_op',
            {'_xi', '_yi'},
            {'_out'},
            '_out = _xi * _yi'
        )

        me.add_in_connector('IN_x')
        me.add_in_connector('IN_y')
        me.add_out_connector('OUT_x')
        me.add_out_connector('OUT_y')
        mx.add_in_connector('IN_result')
        mx.add_out_connector('OUT_result')

        nstate.add_edge(x_node, None, me, 'IN_x',
                        dace.Memlet(f'_x[0:{N}]'))
        nstate.add_edge(y_node, None, me, 'IN_y',
                        dace.Memlet(f'_y[0:{N}]'))
        nstate.add_edge(me, 'OUT_x', tasklet, '_xi',
                        dace.Memlet('_x[i]'))
        nstate.add_edge(me, 'OUT_y', tasklet, '_yi',
                        dace.Memlet('_y[i]'))
        nstate.add_edge(tasklet, '_out', mx, 'IN_result',
                        dace.Memlet(data='_result', subset='0',
                                    wcr='lambda a, b: a + b'))
        nstate.add_edge(mx, 'OUT_result', r_node, None,
                        dace.Memlet('_result[0]'))

        return nsdfg
```

**Key points about the expansion:**
- `@dace.library.expansion` registers this class as a valid expansion
- `expansion()` receives the node (for parameters like `node.n`) and returns
  a **nested SDFG** — DaCe substitutes this in place of the hexagon
- `environments = []` means no external headers or libraries needed
- The internal SDFG uses the same manual construction patterns from Part 1

---

### Part B: The Accelerator Expansion

```python
@dace.library.expansion
class ExpandMACAccelerator(ExpandTransformation):
    """
    Accelerator expansion — emits MMIO calls to your CVA6 accelerator.
    Replace 0xDEADBEEF with your actual hardware register address.
    """
    environments = []

    @staticmethod
    def expansion(node, state, sdfg):
        N = node.n

        nsdfg = dace.SDFG('mac_accelerator')
        nsdfg.add_array('_x', shape=[N], dtype=dace.float64)
        nsdfg.add_array('_y', shape=[N], dtype=dace.float64)
        nsdfg.add_array('_result', shape=[1], dtype=dace.float64)

        nstate = nsdfg.add_state('mac_accel')

        x_node = nstate.add_read('_x')
        y_node = nstate.add_read('_y')
        r_node = nstate.add_write('_result')

        # Single tasklet emitting raw C++ MMIO code
        tasklet = nstate.add_tasklet(
            'accel_call',
            {'_x_ptr', '_y_ptr'},
            {'_result_ptr'},
            # This string is emitted VERBATIM into generated C++
            # Replace with your actual CVA6 MMIO register addresses
            f'''
            volatile uint64_t* accel =
                (volatile uint64_t*)0xDEADBEEF;
            accel[0] = (uint64_t)_x_ptr;      // input X address
            accel[1] = (uint64_t)_y_ptr;      // input Y address
            accel[2] = (uint64_t){N};          // vector length
            accel[3] = 1;                      // start signal
            while(accel[4] != 1);              // poll until done
            *_result_ptr = *(double*)accel[5]; // read result
            ''',
            language=dace.dtypes.Language.CPP
        )

        nstate.add_edge(x_node, None, tasklet, '_x_ptr',
                        dace.Memlet(f'_x[0:{N}]'))
        nstate.add_edge(y_node, None, tasklet, '_y_ptr',
                        dace.Memlet(f'_y[0:{N}]'))
        nstate.add_edge(tasklet, '_result_ptr', r_node, None,
                        dace.Memlet('_result[0]'))

        return nsdfg
```

**Key points about the accelerator expansion:**
- `language=dace.dtypes.Language.CPP` tells DaCe to emit the string **verbatim**
  into the generated C++ file — no Python-to-C translation
- When cross-compiled with `riscv64-unknown-elf-g++` and run on CVA6, those
  MMIO writes go directly to your accelerator's hardware registers
- The tasklet code string is your **escape hatch** from DaCe's abstraction
  into raw hardware control
- Three supported languages: `Python` (default, translated), `CPP` (verbatim),
  `CUDA` (for GPU kernels)

---

### Part C: The Library Node Definition

```python
@dace.library.node
class MACNode(dace.nodes.LibraryNode):
    """
    Multiply-Accumulate: result = sum(x * y)
    Supports multiple implementations for design space exploration.
    """

    # Registry of available implementations
    implementations = {
        'pure':        ExpandMACPure,
        'accelerator': ExpandMACAccelerator
    }
    default_implementation = 'pure'

    # Node parameters — travel with the node in the SDFG,
    # serialized when saved, restored when loaded
    n = dace.properties.SymbolicProperty(default=1)

    def __init__(self, n, *args, **kwargs):
        super().__init__('MAC', *args, **kwargs)
        self.n = n
        # Declare the node's interface connectors
        self.add_in_connector('_x')
        self.add_in_connector('_y')
        self.add_out_connector('_result')
```

**Key points about the node definition:**
- `@dace.library.node` registers this in DaCe's node registry
- `implementations` dict is the **menu of implementations** — adding a new
  accelerator variant is just adding a new entry
- `SymbolicProperty` allows the parameter to be a DaCe symbol, not just a
  concrete integer
- `add_in_connector`/`add_out_connector` in `__init__` declare the ports
  that edges connect to — same mechanism as MapEntry connectors from Part 1

---

### Part D: Using the Library Node

```python
def build_mac_sdfg(implementation='pure'):
    N = 1024

    sdfg = dace.SDFG('mac_test')
    sdfg.add_array('x',      shape=[N], dtype=dace.float64)
    sdfg.add_array('y',      shape=[N], dtype=dace.float64)
    sdfg.add_array('result', shape=[1], dtype=dace.float64)

    state = sdfg.add_state('compute')

    x_node = state.add_read('x')
    y_node = state.add_read('y')
    r_node = state.add_write('result')

    # Instantiate the Library Node
    mac_node = MACNode(n=N)
    mac_node.implementation = implementation  # swap here
    state.add_node(mac_node)

    # Connect using the node's declared connectors
    state.add_edge(x_node, None, mac_node, '_x',
                   dace.Memlet(f'x[0:{N}]'))
    state.add_edge(y_node, None, mac_node, '_y',
                   dace.Memlet(f'y[0:{N}]'))
    state.add_edge(mac_node, '_result', r_node, None,
                   dace.Memlet('result[0]'))

    # Save unexpanded — shows the hexagon in the viewer
    sdfg.save(f'mac_{implementation}_unexpanded.sdfg')

    # Expand before compiling — replaces hexagon with nested SDFG
    sdfg.expand_library_nodes()
    sdfg.validate()
    sdfg.save(f'mac_{implementation}_expanded.sdfg')

    return sdfg


# Test pure implementation
import numpy as np

sdfg = build_mac_sdfg('pure')
compiled = sdfg.compile()

x = np.random.rand(1024)
y = np.random.rand(1024)
result = np.zeros(1)

compiled(x=x, y=y, result=result)
expected = np.dot(x, y)
print(f"Result:   {result[0]:.6f}")
print(f"Expected: {expected:.6f}")
print(f"Error:    {abs(result[0] - expected):.2e}")
```

> **Why the error is ~1e-13 (not zero):**
> Floating point arithmetic is not associative — the order of accumulation
> differs between your pure expansion and `np.dot`. The error is at machine
> epsilon ($\approx 10^{-15}$ for float64 scaled by N operations). This is
> correct behavior, not a bug.

---

### Exercise 5.3: Library Nodes

**Q1.** Open `mac_pure_unexpanded.sdfg` and `mac_pure_expanded.sdfg` side by
side. What is the visual difference?

<details>
<summary>Expected answer</summary>

- **Unexpanded:** A single hexagon node labeled `MAC` in the state — the
  abstract operation. This is the LibraryNode before expansion.
- **Expanded:** The hexagon is replaced by a `NestedSDFG` node containing
  the full map + tasklet structure from `ExpandMACPure.expansion()`.

The hexagon is DaCe's placeholder for "I know what this operation is, but
I haven't decided how to implement it yet." Expansion makes the decision.

</details>

---

**Q2.** What is the minimum change needed to switch from the pure implementation
to the accelerator implementation?

<details>
<summary>Expected answer</summary>

One line:

```python
mac_node.implementation = 'accelerator'
```

The SDFG structure — the surrounding arrays, states, and edges — remains
completely unchanged. Only the expansion changes. This is the power of the
Library Node abstraction for design space exploration.

</details>

---

**Q3.** What does `environments = []` mean and when would you put something
there?

<details>
<summary>Expected answer</summary>

`environments = []` means the expansion has no external dependencies — no
headers to include, no libraries to link. If your expansion calls an external
library, you'd register it:

```python
@dace.library.environment
class CBLASEnvironment:
    cmake_minimum_required = "3.0"
    headers = ["cblas.h"]
    libraries = ["blas"]
    # ... etc

@dace.library.expansion
class ExpandMACBLAS(ExpandTransformation):
    environments = [CBLASEnvironment]  # ← link BLAS
    ...
```

DaCe's build system reads the environment and automatically adds the correct
`#include` directives and linker flags to the generated code.

</details>

---

## 5.4 Accelerator Design Patterns

### Pattern 1: Granularity decision

The right granularity for a Library Node depends on where scheduling decisions
live:

| Granularity | Library Node design | Use when |
|---|---|---|
| Fine-grained | One node per operation (GEMM, Conv, Vector) | DaCe controls dispatch; easier to profile |
| Coarse-grained | One node for whole module | Hardware controls internal dispatch |
| Hierarchical | Outer node expands to inner nodes | Best of both — profile finely, deploy coarsely |

For your research, **fine-grained during Phases 1-3** (profiling and DSE),
**coarse-grained in Phase 5** (hardware implementation).

### Pattern 2: Hierarchical Library Nodes

```python
class CVAcceleratorNode(dace.nodes.LibraryNode):
    implementations = {
        'pure':     ExpandCVPure,      # expands into sub-nodes
        'hardware': ExpandCVHardware   # single MMIO sequence
    }

class ExpandCVPure(ExpandTransformation):
    @staticmethod
    def expansion(node, state, sdfg):
        nsdfg = dace.SDFG('cv_pure')
        # Contains GEMMNode, ConvNode, VectorNode
        # Each independently expandable and profilable
        conv_node = ConvNode(...)
        gemm_node = GEMMNode(...)
        # wire them together
        return nsdfg
```

From the outside: one `CVAcceleratorNode`. In pure mode: a structured graph of
sub-operations you can profile individually. In hardware mode: a single MMIO
sequence to your physical accelerator.

### Pattern 3: Workload-specific recommendations

| Workload | Recommended granularity |
|---|---|
| JEPA | Fine-grained — conv and attention are separate hardware blocks |
| VLM | Mixed — vision fine-grained, language GEMM can be coarse |
| NeSy | Two-level — neural part fine-grained, symbolic part one coarse CPU node |

NeSy is special: the symbolic reasoning component is irregular enough that it
should **not** be a Library Node at all. It stays as native CPU code, and the
heterogeneity argument comes from showing that neural and symbolic parts need
fundamentally different hardware.

---

### Exercise 5.4: Design Patterns

**Q1.** Your self-attention profiling showed GEMMs at 8.0 FLOPs/byte and
softmax pointwise ops at 0.0625 FLOPs/byte. What does this suggest about
Library Node granularity for a `SelfAttentionNode`?

<details>
<summary>Expected answer</summary>

The 128× intensity difference means GEMM and softmax need fundamentally
different hardware:
- GEMM → systolic array (high compute density)
- Softmax pointwise → vector unit (high memory bandwidth)

This argues for **fine-grained Library Nodes** at the attention level:

```python
class SelfAttentionNode(dace.nodes.LibraryNode):
    implementations = {
        'pure':     ExpandSelfAttentionPure,  # expands to sub-nodes
        'hardware': ExpandSelfAttentionHW     # dispatch to accelerator
    }

# Pure expansion contains:
class ExpandSelfAttentionPure(ExpandTransformation):
    def expansion(...):
        # GEMMNode (QK^T) → systolic array
        # SoftmaxNode     → vector unit
        # GEMMNode (AV)   → systolic array
```

A coarse `SelfAttentionNode` that treats the whole operation as one black box
loses the ability to route GEMM vs softmax to different hardware.

</details>

---

**Q2.** How does the Library Node pattern connect to Phase 3 of your
methodology (Accelerator Design Space Exploration)?

<details>
<summary>Expected answer</summary>

Phase 3 involves exploring a family of candidate accelerator configurations.
The Library Node pattern makes this systematic:

```python
# Exploration loop — swap implementations, profile, compare
for impl in ['pure', 'systolic_32', 'systolic_64', 'vector_unit']:
    mac_node.implementation = impl
    sdfg.expand_library_nodes()
    compiled = sdfg.compile()
    # run and profile
    results[impl] = measure_performance(compiled)

# Pick best configuration
best = max(results, key=lambda k: results[k]['gflops'])
```

The SDFG structure stays constant — only the expansion changes. This means
you can explore hundreds of configurations without restructuring your workload
model.

</details>

---

## 5.5 Complete Example: MAC with Both Implementations

```python
import dace
import numpy as np
from dace.transformation.transformation import ExpandTransformation

# ── Expansions ──────────────────────────────────────────────────────

@dace.library.expansion
class ExpandMACPure(ExpandTransformation):
    environments = []
    @staticmethod
    def expansion(node, state, sdfg):
        N = node.n
        nsdfg = dace.SDFG('mac_pure')
        nsdfg.add_array('_x', shape=[N], dtype=dace.float64)
        nsdfg.add_array('_y', shape=[N], dtype=dace.float64)
        nsdfg.add_array('_result', shape=[1], dtype=dace.float64)
        nstate = nsdfg.add_state('mac_compute')
        x_node = nstate.add_read('_x')
        y_node = nstate.add_read('_y')
        r_node = nstate.add_write('_result')
        me, mx = nstate.add_map('mac_map', {'i': f'0:{N}'})
        tasklet = nstate.add_tasklet('mac_op',
            {'_xi', '_yi'}, {'_out'}, '_out = _xi * _yi')
        me.add_in_connector('IN_x');  me.add_in_connector('IN_y')
        me.add_out_connector('OUT_x'); me.add_out_connector('OUT_y')
        mx.add_in_connector('IN_result')
        mx.add_out_connector('OUT_result')
        nstate.add_edge(x_node, None, me, 'IN_x',
                        dace.Memlet(f'_x[0:{N}]'))
        nstate.add_edge(y_node, None, me, 'IN_y',
                        dace.Memlet(f'_y[0:{N}]'))
        nstate.add_edge(me, 'OUT_x', tasklet, '_xi',
                        dace.Memlet('_x[i]'))
        nstate.add_edge(me, 'OUT_y', tasklet, '_yi',
                        dace.Memlet('_y[i]'))
        nstate.add_edge(tasklet, '_out', mx, 'IN_result',
                        dace.Memlet(data='_result', subset='0',
                                    wcr='lambda a, b: a + b'))
        nstate.add_edge(mx, 'OUT_result', r_node, None,
                        dace.Memlet('_result[0]'))
        return nsdfg

@dace.library.expansion
class ExpandMACAccelerator(ExpandTransformation):
    environments = []
    @staticmethod
    def expansion(node, state, sdfg):
        N = node.n
        nsdfg = dace.SDFG('mac_accelerator')
        nsdfg.add_array('_x', shape=[N], dtype=dace.float64)
        nsdfg.add_array('_y', shape=[N], dtype=dace.float64)
        nsdfg.add_array('_result', shape=[1], dtype=dace.float64)
        nstate = nsdfg.add_state('mac_accel')
        x_node = nstate.add_read('_x')
        y_node = nstate.add_read('_y')
        r_node = nstate.add_write('_result')
        tasklet = nstate.add_tasklet(
            'accel_call',
            {'_x_ptr', '_y_ptr'},
            {'_result_ptr'},
            f'''
            volatile uint64_t* accel =
                (volatile uint64_t*)0xDEADBEEF;
            accel[0] = (uint64_t)_x_ptr;
            accel[1] = (uint64_t)_y_ptr;
            accel[2] = (uint64_t){N};
            accel[3] = 1;
            while(accel[4] != 1);
            *_result_ptr = *(double*)accel[5];
            ''',
            language=dace.dtypes.Language.CPP
        )
        nstate.add_edge(x_node, None, tasklet, '_x_ptr',
                        dace.Memlet(f'_x[0:{N}]'))
        nstate.add_edge(y_node, None, tasklet, '_y_ptr',
                        dace.Memlet(f'_y[0:{N}]'))
        nstate.add_edge(tasklet, '_result_ptr', r_node, None,
                        dace.Memlet('_result[0]'))
        return nsdfg

# ── Node definition ─────────────────────────────────────────────────

@dace.library.node
class MACNode(dace.nodes.LibraryNode):
    implementations = {
        'pure':        ExpandMACPure,
        'accelerator': ExpandMACAccelerator
    }
    default_implementation = 'pure'
    n = dace.properties.SymbolicProperty(default=1)
    def __init__(self, n, *args, **kwargs):
        super().__init__('MAC', *args, **kwargs)
        self.n = n
        self.add_in_connector('_x')
        self.add_in_connector('_y')
        self.add_out_connector('_result')

# ── Usage ────────────────────────────────────────────────────────────

def test_mac(implementation='pure'):
    N = 1024
    sdfg = dace.SDFG('mac_test')
    sdfg.add_array('x',      shape=[N], dtype=dace.float64)
    sdfg.add_array('y',      shape=[N], dtype=dace.float64)
    sdfg.add_array('result', shape=[1], dtype=dace.float64)
    state = sdfg.add_state('compute')
    x_node = state.add_read('x')
    y_node = state.add_read('y')
    r_node = state.add_write('result')
    mac_node = MACNode(n=N)
    mac_node.implementation = implementation
    state.add_node(mac_node)
    state.add_edge(x_node, None, mac_node, '_x',
                   dace.Memlet(f'x[0:{N}]'))
    state.add_edge(y_node, None, mac_node, '_y',
                   dace.Memlet(f'y[0:{N}]'))
    state.add_edge(mac_node, '_result', r_node, None,
                   dace.Memlet('result[0]'))
    sdfg.save(f'mac_{implementation}_unexpanded.sdfg')
    sdfg.expand_library_nodes()
    sdfg.validate()
    sdfg.save(f'mac_{implementation}_expanded.sdfg')
    return sdfg

# Test
sdfg = test_mac('pure')
compiled = sdfg.compile()
x = np.random.rand(1024)
y = np.random.rand(1024)
result = np.zeros(1)
compiled(x=x, y=y, result=result)
expected = np.dot(x, y)
print(f"Error: {abs(result[0] - expected):.2e}")
```

We can also allow ourselves to view the `accelerator` argument without having to compile. Simply add the following:

```python
def test_mac_view_only(implementation='accelerator'):
    N = 1024
    sdfg = dace.SDFG('mac_test')
    sdfg.add_array('x',      shape=[N], dtype=dace.float64)
    sdfg.add_array('y',      shape=[N], dtype=dace.float64)
    sdfg.add_array('result', shape=[1], dtype=dace.float64)

    state = sdfg.add_state('compute')
    x_node = state.add_read('x')
    y_node = state.add_read('y')
    r_node = state.add_write('result')

    mac_node = MACNode(n=N)
    mac_node.implementation = implementation
    state.add_node(mac_node)

    state.add_edge(x_node, None, mac_node, '_x',
                   dace.Memlet(f'x[0:{N}]'))
    state.add_edge(y_node, None, mac_node, '_y',
                   dace.Memlet(f'y[0:{N}]'))
    state.add_edge(mac_node, '_result', r_node, None,
                   dace.Memlet('result[0]'))

    # Save unexpanded — hexagon view
    sdfg.save(f'mac_{implementation}_unexpanded.sdfg')

    # Expand — replaces hexagon with MMIO nested SDFG
    sdfg.expand_library_nodes()
    sdfg.save(f'mac_{implementation}_expanded.sdfg')

    # View only — no compile, no validate
    sdfg.view()
    return sdfg

test_mac_view_only('accelerator')
```

---

## Summary: Part 5

You now understand the complete Library Node pattern:

```
@dace.library.expansion          @dace.library.node
class ExpandMyKernelPure:        class MyKernelNode:
    environments = []                implementations = {
    def expansion(...):                  'pure': ExpandMyKernelPure,
        return nested_sdfg               'accel': ExpandMyKernelAccel
                                     }
@dace.library.expansion              n = SymbolicProperty(...)
class ExpandMyKernelAccel:           def __init__(self, n):
    environments = []                    self.add_in_connector(...)
    def expansion(...):                  self.add_out_connector(...)
        # emit MMIO C++ code
        return nested_sdfg
```

**Key concepts:**

| Concept | Details |
|---|---|
| `expansion()` returns a nested SDFG | Substituted in place of the hexagon node |
| `node.n` accesses parameters | Set on the node, available in expansion |
| `language=CPP` | Tasklet code emitted verbatim into generated C++ |
| One-line implementation swap | `mac_node.implementation = 'accelerator'` |
| Connector interface | Declared in `__init__`, same as MapEntry connectors |
| Float error ~1e-13 | Normal — floating point associativity, not a bug |

**The four-layer routine for your methodology:**

```
1. ExpandMyKernelPure      → correctness reference
2. ExpandMyKernelCPUBLAS   → CPU baseline performance
3. ExpandMyKernelSystolic  → accelerator model (gem5-Aladdin)
4. ExpandMyKernelCVA6MMIO  → physical hardware target
```

**Next:** Part 6 covers multi-state control flow — conditionals, loops, and
the NeSy heterogeneity pattern where neural inference and symbolic reasoning
must be dispatched to different hardware.