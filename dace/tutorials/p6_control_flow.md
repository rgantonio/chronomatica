# DaCe Mastery Tutorial
## Part 6: Multi-State Control Flow

> **Prerequisites:** Complete Parts 1–5. You should understand states,
> interstate edges conceptually, and be comfortable with manual SDFG
> construction.

---

## 6.1 Beyond Single-State SDFGs

Everything in Parts 1–5 lived in a **single state**. Real programs have
branches, loops, and conditional execution. In DaCe these are expressed as
**interstate edges** connecting multiple states.

Recall from Part 1: interstate edges are completely separate from memlets.

| Call | Returns |
|---|---|
| `state.edges()` | Memlets inside a state (dataflow) |
| `sdfg.edges()` | Interstate edges between states (control flow) |

### The anatomy of an interstate edge

An interstate edge has two components:

**1. A condition** — a boolean expression evaluated to decide if this edge
is taken:

```python
'flag > 0'      # simple comparison
'i < N'         # loop guard
'1'             # unconditional — always taken (DaCe's representation of no condition)
''              # also unconditional
```

**2. Assignments** — variable updates that happen when the edge is taken:

```python
{'i': '0'}        # initialize loop counter
{'i': 'i + 1'}    # increment loop counter
{}                # no assignments
```

### Interstate variables

Loop counters and branch variables live **on interstate edges**, not inside
any state. They are neither arrays nor DaCe symbols in the traditional sense —
they are **interstate variables** that exist only between states.

| Lives in | Type | Accessed by |
|---|---|---|
| States | Arrays, transients | AccessNodes, Memlets |
| Interstate edges | Loop counters, flags | Conditions, Assignments |

This separation is clean: data computation happens inside states, control
flow happens between states.

---

## 6.2 Conditional SDFGs

### Generating a conditional SDFG

```python
import dace
import numpy as np

N = dace.symbol('N')

@dace.program
def conditional_scale(x: dace.float64[N],
                      flag: dace.int32):
    if flag > 0:
        return x * 2.0
    else:
        return x * 0.5

sdfg = conditional_scale.to_sdfg()
sdfg.simplify()
sdfg.save('conditional.sdfg')
```

---

### Exercise 6.2a: Reading a Conditional SDFG

Open `conditional.sdfg` and answer:

**Q1.** How many states do you see and what is the role of each?

<details>
<summary>Expected answer</summary>

Three states:

- `if_N_guard` — empty guard state with no computation. Contains only the
  branching logic. Named after the line number of the `if` statement.
- `if_N_body_...` — the `flag > 0` branch: map + tasklet computing `x * 2.0`
- `if_N_else_...` — the `not(flag > 0)` branch: map + tasklet computing `x * 0.5`

The guard state is **structurally empty** — no AccessNodes, no maps, no
tasklets. Empty states are valid in DaCe and exist purely to hold branching
logic.

</details>

---

**Q2.** Do the two branches merge back into a single state?

<details>
<summary>Expected answer</summary>

No — because both branches immediately return. Each branch terminates
independently at its own `__return` AccessNode. A merge state only appears
when both branches continue to the same subsequent computation after the
branch. Since there is no computation after the `if/else` here, no merge
state is needed.

</details>

---

**Q3.** What conditions appear on the interstate edges?

<details>
<summary>Expected answer</summary>

Two edges out of the guard state:
- `(flag > 0)` → body branch
- `(not (flag > 0))` → else branch

DaCe automatically negates the condition for the else branch — you only write
one condition in Python and DaCe generates both. The assignments on both edges
are `{}` — no variables are updated at the branch point.

</details>

---

### Exercise 6.2b: Programmatic Inspection

```python
import dace

sdfg = dace.SDFG.from_file('conditional.sdfg')

print("=== States ===")
for state in sdfg.states():
    print(f"State: {state.label}")
    print(f"  nodes: {[type(n).__name__ for n in state.nodes()]}")

print("\n=== Interstate Edges ===")
for edge in sdfg.edges():
    print(f"{edge.src.label} → {edge.dst.label}")
    print(f"  condition:   {edge.data.condition.as_string}")
    print(f"  assignments: {edge.data.assignments}")
    print()
```

**Q4.** Which state has two outgoing interstate edges?

<details>
<summary>Expected answer</summary>

The guard state (`if_N_guard`) has two outgoing edges — one for each branch.
The compute states (`if_N_body_...` and `if_N_else_...`) each have zero
outgoing edges since they terminate at `__return`.

</details>

---

### Building a Conditional SDFG by Hand

```python
import dace
import numpy as np

N = dace.symbol('N')
sdfg = dace.SDFG('conditional_manual')

sdfg.add_array('x',      shape=[N], dtype=dace.float64)
sdfg.add_array('result', shape=[N], dtype=dace.float64)
sdfg.add_scalar('flag', dtype=dace.int32, transient=False)

# ── States ──────────────────────────────────────────────────────────
guard  = sdfg.add_state('guard')      # no computation — pure branching
s_pos  = sdfg.add_state('scale_pos')  # flag > 0 branch
s_neg  = sdfg.add_state('scale_neg')  # flag <= 0 branch
merge  = sdfg.add_state('merge')      # rejoin — no computation

# ── Interstate edges ─────────────────────────────────────────────────
sdfg.add_edge(guard, s_pos,
              dace.InterstateEdge(condition='flag > 0'))
sdfg.add_edge(guard, s_neg,
              dace.InterstateEdge(condition='flag <= 0'))
sdfg.add_edge(s_pos, merge,
              dace.InterstateEdge())   # unconditional
sdfg.add_edge(s_neg, merge,
              dace.InterstateEdge())   # unconditional

# ── s_pos: result[i] = x[i] * 2.0 ──────────────────────────────────
x_node_pos      = s_pos.add_read('x')
result_node_pos = s_pos.add_write('result')
me_pos, mx_pos  = s_pos.add_map('map_pos', {'i': '0:N'})

me_pos.add_in_connector('IN_x');   me_pos.add_out_connector('OUT_x')
mx_pos.add_in_connector('IN_result'); mx_pos.add_out_connector('OUT_result')

tasklet_pos = s_pos.add_tasklet('scale_pos', {'_x'}, {'_result'},
                                '_result = _x * 2.0')

s_pos.add_edge(x_node_pos,    None,      me_pos,      'IN_x',
               dace.Memlet('x[0:N]'))
s_pos.add_edge(me_pos,        'OUT_x',   tasklet_pos, '_x',
               dace.Memlet('x[i]'))
s_pos.add_edge(tasklet_pos,   '_result', mx_pos,      'IN_result',
               dace.Memlet('result[i]'))
s_pos.add_edge(mx_pos,        'OUT_result', result_node_pos, None,
               dace.Memlet('result[0:N]'))

# ── s_neg: result[i] = x[i] * 0.5 ──────────────────────────────────
x_node_neg      = s_neg.add_read('x')
result_node_neg = s_neg.add_write('result')
me_neg, mx_neg  = s_neg.add_map('map_neg', {'i': '0:N'})

me_neg.add_in_connector('IN_x');   me_neg.add_out_connector('OUT_x')
mx_neg.add_in_connector('IN_result'); mx_neg.add_out_connector('OUT_result')

tasklet_neg = s_neg.add_tasklet('scale_neg', {'_x'}, {'_result'},
                                '_result = _x * 0.5')

s_neg.add_edge(x_node_neg,    None,      me_neg,      'IN_x',
               dace.Memlet('x[0:N]'))
s_neg.add_edge(me_neg,        'OUT_x',   tasklet_neg, '_x',
               dace.Memlet('x[i]'))
s_neg.add_edge(tasklet_neg,   '_result', mx_neg,      'IN_result',
               dace.Memlet('result[i]'))
s_neg.add_edge(mx_neg,        'OUT_result', result_node_neg, None,
               dace.Memlet('result[0:N]'))

sdfg.validate()
sdfg.save('conditional_manual.sdfg')

# Test both branches
compiled = sdfg.compile()
x = np.random.rand(1024)

result_pos = np.zeros(1024)
compiled(x=x, result=result_pos, flag=np.int32(1), N=1024)
print("flag=1  max error:", np.max(np.abs(result_pos - x * 2.0)))

result_neg = np.zeros(1024)
compiled(x=x, result=result_neg, flag=np.int32(-1), N=1024)
print("flag=-1 max error:", np.max(np.abs(result_neg - x * 0.5)))
```

---

## 6.3 Loop SDFGs

A loop in DaCe is expressed as a **cycle** in the state machine — a back-edge
from the loop body back to the guard state. The back-edge carries the
**increment assignment** on the loop counter.

### The loop state structure

```
init_state       (sets i = 0 via interstate edge assignment)
     ↓
loop_guard  ←──────────────────────────┐
     │ condition: (i < N)              │
     ↓                                 │
loop_body                              │
     │ assignment: {i: 'i + 1'} ───────┘
     │ condition: (i >= N)
     ↓
after_loop
```

The back-edge from `loop_body → loop_guard` is what makes it a loop.
The assignment `{'i': 'i + 1'}` on that edge advances the counter.

### Generating a loop SDFG

```python
import dace
import numpy as np

N = dace.symbol('N')

@dace.program
def loop_sum(x: dace.float64[N]):
    result = np.float64(0)
    for i in range(N):
        result += x[i]
    return result

sdfg = loop_sum.to_sdfg()
sdfg.simplify()
sdfg.save('loop_sum.sdfg')
```

### Reading loop SDFGs programmatically

```python
import dace

sdfg = dace.SDFG.from_file('loop_sum.sdfg')

print("=== Interstate Edges ===")
for edge in sdfg.edges():
    print(f"{edge.src.label} → {edge.dst.label}")
    print(f"  condition:   {edge.data.condition.as_string}")
    print(f"  assignments: {edge.data.assignments}")
    print()
```

### Reading the output

You will see four interstate edges:

```
call_N → for_N_guard        condition=1       assignments={'i': '0'}
for_N_guard → for_N_body    condition=(i < N) assignments={}
for_N_guard → assign_N_M    condition=(not (i < N)) assignments={}
for_N_body → for_N_guard    condition=1       assignments={'i': '(i + 1)'}
```

The key edges:
- `condition=1` means **unconditional** — always taken
- `assignments={'i': '0'}` initializes the loop counter
- `assignments={'i': '(i + 1)'}` is the **back-edge increment** — this is
  what makes it a loop

---

### Exercise 6.3: Reading Loop SDFGs

**Q1.** Where is the loop counter `i` declared — is it an array or a symbol?

<details>
<summary>Expected answer</summary>

`i` is an **interstate variable** — it exists only on interstate edges.
It is neither an array (no AccessNode) nor a DaCe symbol in the traditional
sense. DaCe creates it implicitly when it sees `{'i': '0'}` on the first edge.

Think of it as a register that lives between states — written by edge
assignments and read by edge conditions, but never appearing inside a
state's dataflow graph.

</details>

---

**Q2.** Which edge is the back-edge that makes this a loop?

<details>
<summary>Expected answer</summary>

`for_N_body → for_N_guard` with `assignments={'i': '(i + 1)'}`. This edge
creates the cycle in the state graph. Without this back-edge it would be
a linear chain of states, not a loop.

</details>

---

### Building a Loop SDFG by Hand

```python
import dace
import numpy as np

N = dace.symbol('N')
sdfg = dace.SDFG('loop_sum_manual')

sdfg.add_array('x',      shape=[N], dtype=dace.float64)
sdfg.add_array('result', shape=[1], dtype=dace.float64)
sdfg.add_symbol('i', dace.int32)  # loop counter as interstate variable

# ── States ──────────────────────────────────────────────────────────
init  = sdfg.add_state('init')
guard = sdfg.add_state('loop_guard')
body  = sdfg.add_state('loop_body')
after = sdfg.add_state('after_loop')

# ── Interstate edges ─────────────────────────────────────────────────
sdfg.add_edge(init,  guard,
              dace.InterstateEdge(assignments={'i': '0'}))
sdfg.add_edge(guard, body,
              dace.InterstateEdge(condition='i < N'))
sdfg.add_edge(guard, after,
              dace.InterstateEdge(condition='i >= N'))
sdfg.add_edge(body,  guard,                          # ← back-edge
              dace.InterstateEdge(assignments={'i': 'i + 1'}))

# ── init: result[0] = 0 ─────────────────────────────────────────────
result_init    = init.add_write('result')
init_tasklet   = init.add_tasklet('init_result', {}, {'_out'},
                                  '_out = 0')
init.add_edge(init_tasklet, '_out', result_init, None,
              dace.Memlet('result[0]'))

# ── loop body: result[0] += x[i] ────────────────────────────────────
# result is both read AND written — needs two separate AccessNodes
x_read      = body.add_read('x')
result_read  = body.add_read('result')    # reads old value
result_write = body.add_write('result')   # writes new value

body_tasklet = body.add_tasklet(
    'update_result',
    {'_x', '_result_in'},
    {'_result_out'},
    '_result_out = _result_in + _x'
)

body.add_edge(x_read,       None,          body_tasklet, '_x',
              dace.Memlet('x[i]'))
body.add_edge(result_read,  None,          body_tasklet, '_result_in',
              dace.Memlet('result[0]'))
body.add_edge(body_tasklet, '_result_out', result_write, None,
              dace.Memlet('result[0]'))

sdfg.validate()
sdfg.save('loop_sum_manual.sdfg')

compiled = sdfg.compile()
x = np.random.rand(16)
result = np.zeros(1)
compiled(x=x, result=result, N=16)
print("Result:  ", result[0])
print("Expected:", np.sum(x))
print("Error:   ", abs(result[0] - np.sum(x)))
```

### Key concept: the read/write split

When an array is both read and written in the same state (in-place update),
you need **two separate AccessNodes**:

```python
result_read  = body.add_read('result')    # old value coming in
result_write = body.add_write('result')   # new value going out
```

This makes the dataflow unambiguous:
```
result_read → tasklet → result_write
(old value)              (new value)
```

DaCe cannot represent an in-place update with a single AccessNode because
that would create a cycle in the dataflow graph within a state — which is
not allowed.

---

### Exercise 6.3b: Manual Loop

**Q3.** The `after_loop` state is empty. Why is this valid, and why does the
auto-generated version have an extra assignment state that your manual version
doesn't?

<details>
<summary>Expected answer</summary>

Empty states are valid in DaCe — they exist purely as control flow
waypoints. Your manual `after_loop` has nothing to do because the caller
reads `result` directly after execution.

The auto-generated version adds an `assign_N_M` state that copies
`result → __return`. DaCe is conservative: when a function has a `return`
statement, it creates a separate `__return` array and copies the result into
it rather than assuming the caller's array is safe to use directly.

Your manual version is more efficient — one less memory copy — because you
explicitly declared `result` as the output and guaranteed its availability
to the caller.

</details>

---

## 6.4 Reading Graph Structure Without a Viewer

When analyzing large SDFGs programmatically, you need to reconstruct program
structure from `sdfg.edges()` output alone.

### State naming conventions

| Pattern | Meaning |
|---|---|
| `for_N_guard` | Guard state of a for/while loop at line N |
| `for_N_body` | Body of a for loop |
| `if_N_guard` | Guard state of an if statement at line N |
| `if_N_body_...` | True branch of an if |
| `if_N_else_...` | False branch |
| `call_N` | Function call or initialization at line N |
| `assign_N_M` | Assignment statement at line N |

### The three-step reading process

```
Step 1: sdfg.edges()
        → reconstruct control flow skeleton
        → loops: look for back-edges with i=i+1 assignments
        → branches: two edges with opposite conditions from same state

Step 2: state.nodes() per state
        → MapEntry/MapExit = parallel loop
        → Tasklet = computation leaf
        → AccessNode = memory read/write
        → NestedSDFG = function call / library expansion

Step 3: state.edges() per state
        → memlet subsets tell you access patterns
        → WCR tells you reductions
        → boundary edges (AccessNode↔MapEntry) tell you total data movement
```

### Identifying loops vs branches from edges alone

```python
import dace

sdfg = dace.SDFG.from_file('any.sdfg')

# Detect back-edges (loops)
state_labels = {s.label for s in sdfg.states()}
for edge in sdfg.edges():
    # A back-edge goes to a state that was already "before" in the graph
    if edge.data.assignments:
        print(f"ASSIGNMENT EDGE: {edge.src.label} → {edge.dst.label}")
        print(f"  assigns: {edge.data.assignments}")

# Detect branches (two edges from same source with opposite conditions)
from collections import defaultdict
out_edges = defaultdict(list)
for edge in sdfg.edges():
    out_edges[edge.src.label].append(edge)

for src_label, edges in out_edges.items():
    if len(edges) == 2:
        c0 = edges[0].data.condition.as_string
        c1 = edges[1].data.condition.as_string
        print(f"BRANCH at {src_label}: '{c0}' vs '{c1}'")
```

---

## 6.5 The NeSy Heterogeneity Pattern

This is the most important application of multi-state SDFGs for your
research. NeSy (Neuro-Symbolic) architectures mix:

- **Neural inference** — GPU/accelerator-friendly, high arithmetic intensity
- **Symbolic reasoning** — CPU-bound, irregular memory access, graph
  traversals, logic unification

The control flow pattern looks like:

```
neural_inference_state    (AccessNodes + Maps — accelerator)
          ↓
symbolic_guard            (empty — checks if symbolic is needed)
          ├──→ (needs_symbolic)   symbolic_reasoning_state  (CPU only)
          │                                ↓
          └──→ (not needs_symbolic)        merge_state
                                           ↑
                                 (both branches join here)
```

### Building the NeSy pattern

```python
import dace
import numpy as np

N = dace.symbol('N')
sdfg = dace.SDFG('nesy_dispatch')

sdfg.add_array('features',  shape=[N], dtype=dace.float64)
sdfg.add_array('output',    shape=[N], dtype=dace.float64)
sdfg.add_scalar('confidence', dtype=dace.float64, transient=False)
sdfg.add_scalar('threshold',  dtype=dace.float64, transient=False)

# ── States ──────────────────────────────────────────────────────────
neural_state    = sdfg.add_state('neural_inference')
symbolic_guard  = sdfg.add_state('symbolic_guard')
symbolic_state  = sdfg.add_state('symbolic_reasoning')
merge_state     = sdfg.add_state('merge')

# ── Interstate edges ─────────────────────────────────────────────────
sdfg.add_edge(neural_state,   symbolic_guard,
              dace.InterstateEdge())   # always go to guard after neural
sdfg.add_edge(symbolic_guard, symbolic_state,
              dace.InterstateEdge(condition='confidence < threshold'))
sdfg.add_edge(symbolic_guard, merge_state,
              dace.InterstateEdge(condition='confidence >= threshold'))
sdfg.add_edge(symbolic_state, merge_state,
              dace.InterstateEdge())

# ── neural_inference: scale features (placeholder for real inference)
feat_read  = neural_state.add_read('features')
out_write  = neural_state.add_write('output')
me, mx     = neural_state.add_map('neural_map', {'i': '0:N'})

me.add_in_connector('IN_f');    me.add_out_connector('OUT_f')
mx.add_in_connector('IN_out');  mx.add_out_connector('OUT_out')

neural_tasklet = neural_state.add_tasklet(
    'neural_op', {'_f'}, {'_out'}, '_out = _f * 0.9')

neural_state.add_edge(feat_read,      None,     me,             'IN_f',
                      dace.Memlet('features[0:N]'))
neural_state.add_edge(me,             'OUT_f',  neural_tasklet, '_f',
                      dace.Memlet('features[i]'))
neural_state.add_edge(neural_tasklet, '_out',   mx,             'IN_out',
                      dace.Memlet('output[i]'))
neural_state.add_edge(mx,             'OUT_out', out_write,     None,
                      dace.Memlet('output[0:N]'))

# ── symbolic_reasoning: refine output (placeholder for real symbolic)
feat_read_s = symbolic_state.add_read('features')
out_read_s  = symbolic_state.add_read('output')
out_write_s = symbolic_state.add_write('output')

sym_tasklet = symbolic_state.add_tasklet(
    'symbolic_op',
    {'_f', '_out_in'},
    {'_out'},
    '_out = _out_in + (_f * 0.1)'   # symbolic correction
)

symbolic_state.add_edge(feat_read_s, None,       sym_tasklet, '_f',
                        dace.Memlet('features[0]'))
symbolic_state.add_edge(out_read_s,  None,       sym_tasklet, '_out_in',
                        dace.Memlet('output[0]'))
symbolic_state.add_edge(sym_tasklet, '_out',     out_write_s, None,
                        dace.Memlet('output[0]'))

sdfg.validate()
sdfg.save('nesy_dispatch.sdfg')

compiled = sdfg.compile()
features   = np.random.rand(64)
output     = np.zeros(64)
confidence = np.float64(0.3)   # low confidence → symbolic triggered
threshold  = np.float64(0.5)

compiled(features=features, output=output,
         confidence=confidence, threshold=threshold, N=64)

print("Symbolic path taken (confidence < threshold):",
      confidence < threshold)
print("Output[0]:", output[0])
```

---

### Exercise 6.5: NeSy Pattern

**Q1.** What does the `symbolic_guard` state contain, and why is it empty?

<details>
<summary>Expected answer</summary>

`symbolic_guard` is an empty state — no AccessNodes, no maps, no tasklets.
It exists purely to hold the branching logic: "should we invoke symbolic
reasoning?" The decision is made by comparing `confidence` against `threshold`
on the interstate edges.

This is the standard pattern: whenever you need to make a runtime decision
about which computational path to take, use an empty guard state with
conditional interstate edges. The computation itself lives in the states
that follow.

</details>

---

**Q2.** In this pattern, which states map to which hardware in your
heterogeneous system?

<details>
<summary>Expected answer</summary>

| State | Hardware target | Why |
|---|---|---|
| `neural_inference` | Accelerator (systolic array / GPU) | Regular compute, high arithmetic intensity |
| `symbolic_guard` | CPU (trivial comparison) | Single threshold check |
| `symbolic_reasoning` | CPU only | Irregular memory access, graph traversal, logic unification — not accelerator-friendly |
| `merge` | N/A (empty) | Pure control flow waypoint |

The NeSy heterogeneity argument: the SDFG structure itself tells you that
`symbolic_reasoning` cannot be offloaded — its computation is fundamentally
CPU-bound and cannot be expressed as a regular map over arrays.

</details>

---

**Q3.** How would you extend this pattern for iterative symbolic reasoning
(multiple rounds of refinement)?

<details>
<summary>Expected answer</summary>

Add a back-edge from the symbolic state back to the guard, with an iteration
counter:

```python
sdfg.add_symbol('sym_iter', dace.int32)

# From neural → guard: initialize counter
sdfg.add_edge(neural_state, symbolic_guard,
              dace.InterstateEdge(assignments={'sym_iter': '0'}))

# Back-edge: repeat symbolic up to max_iters times
sdfg.add_edge(symbolic_state, symbolic_guard,
              dace.InterstateEdge(
                  assignments={'sym_iter': 'sym_iter + 1'},
                  condition='sym_iter < max_iters'
              ))

# Exit when converged or max iterations reached
sdfg.add_edge(symbolic_state, merge_state,
              dace.InterstateEdge(
                  condition='sym_iter >= max_iters'
              ))
```

This combines the loop pattern (Section 6.3) with the conditional pattern
(Section 6.2) — exactly what iterative NeSy inference requires.

</details>

---

## 6.6 Summary of State Naming Patterns

When you receive an SDFG from a colleague or from auto-generation and need
to understand it without a viewer, use this reference:

```
call_N           → initialization or function call at line N
for_N_guard      → loop condition check
for_N_body / for_N_slice_x_M  → loop body
if_N_guard       → branch decision point (always empty)
if_N_body_...    → true branch
if_N_else_...    → false branch
assign_N_M       → post-loop or return assignment
```

**Interstate edge patterns:**

```
condition='1', assignments={'i': '0'}     → loop initialization
condition='i < N', assignments={}         → loop entry
condition='not (i < N)', assignments={}   → loop exit
condition='1', assignments={'i': 'i+1'}   → loop back-edge (increment)
condition='flag > 0', assignments={}      → conditional branch
```

---

## Summary: Part 6

You now understand the complete multi-state control flow model:

| Concept | Key detail |
|---|---|
| Guard states | Always empty — hold branching logic only |
| Merge states | Empty — control flow waypoints after branches |
| Unconditional edge | `condition='1'` in DaCe |
| Loop back-edge | Edge with `assignments={'i': 'i+1'}` creating a cycle |
| Interstate variables | Live on edges only — not arrays, not symbols |
| Read/write split | In-place updates need `add_read` + `add_write` for same array |
| `sdfg.edges()` vs `state.edges()` | Interstate edges vs memlets — completely separate |

**The NeSy heterogeneity pattern:**
```
neural_inference (accelerator) → symbolic_guard (empty) →
    ├── symbolic_reasoning (CPU only, irregular)
    └── merge (skip symbolic)
```

The SDFG structure itself reveals the hardware dispatch requirement —
no profiling needed to know that symbolic reasoning cannot be accelerated.

**Next:** Part 7 is the end-to-end case study — self-attention through the
complete analysis pipeline, combining everything from Parts 1–6.