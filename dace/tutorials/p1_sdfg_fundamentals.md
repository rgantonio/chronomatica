# DaCe Mastery Tutorial
## Part 1: SDFG Fundamentals

> **About this tutorial**
> This is a hands-on, exercise-driven guide to mastering DaCe and Stateful Dataflow
> Multigraphs (SDFGs). Each section introduces a concept, shows working code, poses
> questions for you to answer independently, and provides expected answers in dropdowns.
> All fixes and API nuances discovered during development are documented inline.
>
> **Prerequisites:** DaCe installed in your conda environment, VS Code with the
> `phschaad.sdfv` extension for SDFG viewing.
>
> **Setup fix:** The VS Code SDFG viewer backend requires `flask` and `python-dotenv`
> installed in your conda environment. If the viewer fails to launch, run:
> ```bash
> conda install flask python-dotenv
> ```
> Also ensure VS Code's Python interpreter is set to your conda environment.

---

## 1.1 What is an SDFG?

A **Stateful Dataflow Multigraph (SDFG)** is DaCe's core program representation.
It separates a program into two distinct layers:

- **Control flow** — represented as a graph of **states** connected by **interstate edges**
- **Data flow** — represented as a dataflow graph *within* each state

This separation is what makes DaCe powerful for analysis and optimization: you can
reason about data movement independently from control flow.

### The SDFG vocabulary

**Between states — interstate edges:**

| Element | Meaning |
|---|---|
| State | A unit of sequential execution containing a dataflow graph |
| Interstate edge | Control flow transition between states, carrying a condition and optional variable assignments |

**Inside a state — dataflow nodes:**

| Node | Shape in viewer | Meaning |
|---|---|---|
| `AccessNode` | Circle/oval | A memory location — array or scalar |
| `Tasklet` | Octagon | A small unit of computation (a few lines of code) |
| `MapEntry`/`MapExit` | Trapezoid pair | A parallel loop — everything between entry and exit runs for each iteration |
| `NestedSDFG` | Rectangle with border | A whole sub-SDFG inlined at this point |
| `LibraryNode` | Hexagon | An abstract operation with multiple possible implementations |

**Inside a state — edges:**

| Edge | Carries | Meaning |
|---|---|---|
| Memlet | Data subset + volume | A data transfer between nodes |

### Key principle: dataflow, not sequential

Inside a state, nodes can execute in **any order** consistent with data dependencies.
There is no implicit sequencing — only the data edges enforce ordering.
This is fundamentally different from sequential code.

---

## 1.2 Generating Your First SDFG

The simplest way to get an SDFG is from a `@dace.program` function.

```python
import dace
import numpy as np

N = dace.symbol('N')

@dace.program
def axpy(a: dace.float64, x: dace.float64[N], y: dace.float64[N]):
    return a * x + y

sdfg = axpy.to_sdfg()
sdfg.simplify()          # clean up redundant states
sdfg.save('axpy.sdfg')   # open in VS Code SDFG viewer
sdfg.view()              # also opens in browser
```

Open `axpy.sdfg` in the VS Code SDFG viewer and study it before answering the
questions below.

---

### Exercise 1.2: Reading the AXPY SDFG

Answer these questions by reading the SDFG visually — not by reading the source code.

**Q1.** How many memlets enter the multiply tasklet? What subsets do they carry?

<details>
<summary>Expected answer</summary>

Two memlets enter the multiply tasklet: `a[0]` (the scalar) and `x[__i0]`
(one element of x at map iteration `__i0`). Both carry a volume of 1 — one
element per map iteration.

**What is a subset?** The subset is the index expression on a memlet — it describes
*which portion* of an array is being accessed. `x[__i0]` means element `__i0` of x.
This matters for roofline analysis: the subset tells you the per-iteration data access
pattern, and the volume tells you total elements transferred.

</details>

---

**Q2.** Is there a sequential dependency between the multiply and the add, or are
they independent per iteration?

<details>
<summary>Expected answer</summary>

There is a data dependency, but it is **data-driven not control-flow-driven**.
The add tasklet reads `_tmp0`, which is written by the multiply tasklet. In DaCe's
dataflow model, that data edge *is* the dependency — there is no implicit ordering,
only what data edges enforce.

This is an important distinction: sequencing in DaCe is always expressed through
data, never through implicit program order.

</details>

---

**Q3.** What does the interstate edge look like — what condition does it carry?

<details>
<summary>Expected answer</summary>

There are **no interstate edges** in the AXPY SDFG. The program has no control
flow (no if/else, no loops), so DaCe correctly generates a single state with no
transitions between states.

Interstate edges only appear when your program has branching or looping. To see
interstate edges, try a conditional example:

```python
@dace.program
def conditional_axpy(a: dace.float64, x: dace.float64[N],
                     y: dace.float64[N], flag: dace.int32):
    if flag > 0:
        return a * x + y
    else:
        return y
```

This generates a guard state with two outgoing edges: `flag > 0` and
`not(flag > 0)`.

</details>

---

**Q4.** Where does the output array get written — which AccessNode, at which point
in the dataflow?

<details>
<summary>Expected answer</summary>

The output gets written to the `__return` AccessNode. DaCe uses `__return` as the
name for a function's return value when you use `return` in a `@dace.program`.
In your manually-built SDFGs later, you'll name this array whatever you declare
as your output (e.g., `result`).

Note that `__return` is a *write* node — shown differently from read nodes in the
viewer.

</details>

---

## 1.3 Programmatic Inspection

You can inspect any SDFG in code — essential for analyzing large graphs programmatically.

```python
import dace

sdfg = dace.SDFG.from_file('axpy.sdfg')

print(f"SDFG name: {sdfg.name}")
print(f"Symbols: {sdfg.symbols}")
print(f"Arrays: {list(sdfg.arrays.keys())}")
print()

for state in sdfg.states():
    print(f"=== State: {state.label} ===")
    for node in state.nodes():
        print(f"  {type(node).__name__:20s}  {node.label}")
        for edge in state.out_edges(node):
            print(f"    {edge.src.label} → {edge.dst.label}  "
                  f"memlet: {edge.data}  "
                  f"subset: {edge.data.subset}  "
                  f"volume: {edge.data.volume}")
    print()
```

To inspect transient arrays:

```python
for name, arr in sdfg.arrays.items():
    print(f"{name:20s}  transient={arr.transient}  shape={arr.shape}")
```

### Why node order is arbitrary

When you iterate `state.nodes()`, the order does **not** match the visual layout
in the SDFG viewer, and does not match the computational order. This is intentional.

DaCe stores nodes inside a state using a **NetworkX directed graph** internally.
More importantly, DaCe deliberately does not define a sequential execution order
within a state. The whole point of the dataflow model is:

> *"Execute nodes in any order that respects the data edges."*

This is what enables parallelism. If DaCe enforced a fixed order, it couldn't
parallelize across nodes.

**How DaCe finds execution order at compile time:**

When compiling to C++, DaCe performs a **topological sort** of the dataflow graph —
it walks the edges and finds a valid execution order:

```
a, x → MapEntry(_Mult_) → Tasklet(_Mult_) → MapExit → __tmp0
__tmp0, y → MapEntry(_Add_) → Tasklet(_Add_) → MapExit → __return
```

The compiler derives this order by following edges, not by node insertion order.
Any topological ordering that respects data edges is valid — and for parallel
maps, multiple orderings are equivalent.

**Practical implication:** When writing analysis code that walks the graph,
never assume node order. Always follow edges to understand dependencies.

---

### Exercise 1.3: Programmatic Inspection

Run the inspection code on `axpy.sdfg` and answer:

**Q1.** Does the printed node order match the visual order in the SDFG viewer?

<details>
<summary>Expected answer</summary>

No — and this is intentional. Inside a state, DaCe stores nodes as a *set*, not
a sequence. There is no "first" or "last" node in a dataflow graph. The visual
layout is the viewer arranging nodes for readability, not a meaningful ordering.
Never read sequencing from node print order.

</details>

---

**Q2.** Which arrays are transient and which are not? What does transient mean?

<details>
<summary>Expected answer</summary>

Arrays like `__tmp0` are transient (`transient=True`) — they exist only inside
the SDFG and are never exposed as inputs or outputs. Your actual inputs `x`, `y`,
`a` and output `__return` are non-transient (`transient=False`).

**Why transients matter for hardware design:** Transients represent on-chip
buffering. When designing accelerators, transient arrays are candidates for
scratchpad or register file storage rather than going off-chip to DRAM.

</details>

---

**Q3.** What does `__tmp0` represent — did you write it in the Python code?

<details>
<summary>Expected answer</summary>

No — DaCe inserted `__tmp0` automatically as a buffer between the multiply and
add tasklets. It exists because the two operations needed a place to pass data.
This is a common pattern in auto-generated SDFGs: intermediate results become
transient arrays even when they were just temporary values in your Python code.

</details>

---

## 1.4 Building an SDFG by Hand

Building SDFGs manually is the most effective way to deeply understand the model.
This forces you to think explicitly about every node, connector, and edge.

> **Important:** Interstate edges live on `sdfg.edges()`, not on states.
> Memlets live on `state.edges()`. These are completely separate namespaces:
>
> | Call | Returns |
> |---|---|
> | `state.edges()` | Memlets inside a state (dataflow) |
> | `sdfg.edges()` | Interstate edges between states (control flow) |

### Working example: Manual AXPY

```python
import dace
import numpy as np

# 1. Create an empty SDFG
N = dace.symbol('N')
sdfg = dace.SDFG('axpy_manual')

# 2. Declare arrays
sdfg.add_array('x', shape=[N], dtype=dace.float64)
sdfg.add_array('y', shape=[N], dtype=dace.float64)
sdfg.add_array('result', shape=[N], dtype=dace.float64)
sdfg.add_scalar('a', dtype=dace.float64, transient=False)

# 3. Add a state
state = sdfg.add_state('compute')

# 4. Add AccessNodes
x_node   = state.add_read('x')
y_node   = state.add_read('y')
a_node   = state.add_read('a')
out_node = state.add_write('result')

# 5. Add the Map (parallel loop over [0:N])
map_entry, map_exit = state.add_map('parallel_i', {'i': '0:N'})

map_entry.add_in_connector('IN_a')
map_entry.add_in_connector('IN_x')
map_entry.add_in_connector('IN_y')
map_entry.add_out_connector('OUT_a')
map_entry.add_out_connector('OUT_x')
map_entry.add_out_connector('OUT_y')

map_exit.add_in_connector('IN_result')
map_exit.add_out_connector('OUT_result')

# 6. Add a Tasklet
tasklet = state.add_tasklet(
    name='axpy_compute',
    inputs={'_a', '_x', '_y'},
    outputs={'_out'},
    code='_out = _a * _x + _y'
)

# 7. Connect with Memlets
state.add_edge(a_node, None, map_entry, 'IN_a',
               dace.Memlet(data='a', subset='0'))
state.add_edge(x_node, None, map_entry, 'IN_x',
               dace.Memlet('x[0:N]'))
state.add_edge(y_node, None, map_entry, 'IN_y',
               dace.Memlet('y[0:N]'))

state.add_edge(map_entry, 'OUT_a', tasklet, '_a',
               dace.Memlet(data='a', subset='0'))
state.add_edge(map_entry, 'OUT_x', tasklet, '_x',
               dace.Memlet('x[i]'))
state.add_edge(map_entry, 'OUT_y', tasklet, '_y',
               dace.Memlet('y[i]'))

state.add_edge(tasklet, '_out', map_exit, 'IN_result',
               dace.Memlet('result[i]'))
state.add_edge(map_exit, 'OUT_result', out_node, None,
               dace.Memlet('result[0:N]'))

# 8. Validate and save
sdfg.validate()
sdfg.save('axpy_manual.sdfg')

# 9. Compile and test
compiled = sdfg.compile()
a = np.float64(2.0)
x = np.random.rand(1024)
y = np.random.rand(1024)
result = np.zeros(1024)
compiled(a=a, x=x, y=y, result=result, N=1024)

expected = a * x + y
print("Max error:", np.max(np.abs(result - expected)))
```

### Understanding `add_edge` arguments

```python
state.add_edge(src, src_conn, dst, dst_conn, memlet)
```

| Argument | Type | Meaning |
|---|---|---|
| `src` | node | Source node |
| `src_conn` | `str` or `None` | Connector name on source (`None` for AccessNodes) |
| `dst` | node | Destination node |
| `dst_conn` | `str` or `None` | Connector name on destination (`None` for AccessNodes) |
| `memlet` | `dace.Memlet` | What data flows along this edge |

**`None` always appears on the AccessNode side.** AccessNodes have no named
connectors — they are just memory locations. MapEntry/MapExit always need named
connectors. Tasklet connectors match the variable names in the tasklet code.

### Memlet subset scoping rule

The memlet subset must reflect what data is visible at that position in the graph:

| Position | Subset for `x[0:N]` array |
|---|---|
| Outside the map (AccessNode → MapEntry) | `x[0:N]` — full array crosses the boundary |
| Inside the map (MapEntry → Tasklet) | `x[i]` — one element per iteration |

---

### Exercise 1.4: Manual AXPY

Build and run the manual AXPY above, then answer:

**Q1.** Open both `axpy.sdfg` (auto-generated) and `axpy_manual.sdfg` side by side
in the viewer. What is the most significant structural difference?

<details>
<summary>Expected answer</summary>

The auto-generated SDFG has **two separate tasklets** — one for multiply and one
for add — with a `__tmp0` transient buffer between them. Your manual version has
**one fused tasklet** that computes `_out = _a * _x + _y` directly.

Both are correct and compute the same result. Your version is actually more
efficient — it eliminates the intermediate transient and reduces memory traffic.
This is a preview of tasklet fusion transformations: you accidentally wrote the
already-optimized version by hand.

</details>

---

**Q2.** What does `sdfg.validate()` do and why should you always call it?

<details>
<summary>Expected answer</summary>

`sdfg.validate()` runs structural correctness checks on the graph:
- Every connector referenced in an edge exists on the node
- Every memlet subset is valid for the array's shape
- No dangling edges or disconnected nodes
- Data flow is consistent — no array written by two nodes simultaneously without explicit handling

Think of it as a **type checker for your dataflow graph**. It catches structural
errors early as clear Python exceptions rather than cryptic C++ compiler crashes
deep in the generated code. Always call it after building an SDFG manually.

</details>

---

**Q3.** Does the map created by `add_map` always come as a pair? Why?

<details>
<summary>Expected answer</summary>

Yes — `add_map` always returns a matched `(map_entry, map_exit)` pair. A Map in
DaCe is *defined* as a scope with both a boundary entry and exit. They have no
meaning without each other. Everything added between them is inside the parallel
scope. You can never have one without the other.

</details>

---

## 1.5 Common Errors and Fixes Reference

| Error | Cause | Fix |
|---|---|---|
| `Destination connector cannot be None for MapEntry` | Edge to MapEntry has `None` as connector | Add named connector: `map_entry.add_in_connector('IN_x')` |
| `Memlet leading to nonexistent connector IN_x` | Connector not declared before edge | Call `add_in_connector('IN_x')` before `add_edge` |
| `'NoneType' has no attribute 'dims'` | Scalar memlet has `None` subset | Use `dace.Memlet(data='a', subset='0')` not `dace.Memlet('a')` |
| `AttributeError: 'Memlet' has no attribute 'volume'` | API changed in DaCe 1.0.x | Use `m.num_elements` (attribute) not `m.volume()` (method) |
| `cannot determine truth value of Relational` | Comparing symbolic SymPy expression with `>` | Use `!= sympy.Integer(0)` for zero checks |

---

## Summary: Part 1

You now understand the fundamental building blocks of the SDFG model:

- **States** contain dataflow graphs — nodes and memlets
- **Interstate edges** carry control flow — conditions and assignments
- **AccessNodes** represent memory; **Tasklets** represent computation; **Maps** represent parallelism
- **Memlets** annotate every edge with what data flows and which subset
- The subset scoping rule: full array outside a map, indexed element inside
- Connectors must be explicitly declared on MapEntry/MapExit before connecting edges
- Scalars always need explicit subset `'0'` in memlets

**Next:** Part 2 covers matrix multiplication — introducing 2D maps, WCR reductions,
LibraryNodes, and how the `@` operator maps to the SDFG model.