# DaCe Mastery Tutorial
## Part 8: Streams — Pipeline Communication Between Accelerator Stages

> **Prerequisites:** Complete Parts 1–7. You should be comfortable with maps,
> memlets, transient arrays, and the Library Node pattern. This part introduces
> streams — the last major DaCe data concept needed for heterogeneous system
> design.

---

## 8.1 What Problem Do Streams Solve?

Everything in Parts 1–7 used **arrays** as communication buffers between maps.
When two maps communicate through a transient array:

```
Producer map → writes ALL N elements to memory → Consumer map reads ALL N elements
```

The consumer cannot start until the producer finishes entirely. This is
**bulk transfer** — the full array must exist in memory before the next
stage begins.

A **stream** replaces the transient array with a FIFO queue:

```
Producer map → pushes element 0 → Consumer reads element 0
               pushes element 1 → Consumer reads element 1
               pushes element 2 → Consumer reads element 2
               (producer and consumer run simultaneously)
```

No memory round-trip, no waiting for the full array. This is
**pipeline parallelism** — stages overlap in time.

---

## 8.2 Streams vs Arrays — The Full Comparison

| Property | Array | Stream |
|---|---|---|
| Python type | `dace.data.Array` | `dace.data.Stream` |
| Access pattern | Random — any index anytime | Sequential — FIFO push/pop only |
| Always transient | No | Yes — never exposed to caller |
| Hardware maps to | SRAM / DRAM | On-chip FIFO register |
| Parallelism type | Data parallel | Pipeline parallel |
| DaCe declaration | `sdfg.add_array(...)` | `sdfg.add_stream(...)` |
| Schedule required | Any | **Sequential** (see Section 8.4) |

Streams are always transient because they only exist as communication channels
between two stages — they never make sense as external inputs or outputs of
an SDFG.

---

## 8.3 Your First Stream — Two-Step AXPY

Let's convert the two-step AXPY from Part 3 (multiply then add, with a
transient in between) to use a stream instead.

```python
import dace
import numpy as np

N = dace.symbol('N')
sdfg = dace.SDFG('axpy_stream')

# Declare arrays
sdfg.add_array('a',      shape=[1],  dtype=dace.float64,
               transient=False)
sdfg.add_array('x',      shape=[N],  dtype=dace.float64,
               transient=False)
sdfg.add_array('y',      shape=[N],  dtype=dace.float64,
               transient=False)
sdfg.add_array('result', shape=[N],  dtype=dace.float64,
               transient=False)

# Declare a stream — FIFO buffer of float64 elements
# buffer_size=1: single-entry FIFO, fully pipelined
sdfg.add_stream('tmp_stream',
                dtype=dace.float64,
                buffer_size=1,
                transient=True)

state = sdfg.add_state('compute')

# Access nodes
a_node      = state.add_read('a')
x_node      = state.add_read('x')
y_node      = state.add_read('y')
out_node    = state.add_write('result')
stream_node = state.add_access('tmp_stream')

# ── Map 1: multiply — producer, pushes into stream ───────────────────
# CRITICAL: must be Sequential — see Section 8.4
me1, mx1 = state.add_map('multiply_map', {'i': '0:N'},
                         schedule=dace.ScheduleType.Sequential)
mult_tasklet = state.add_tasklet(
    'multiply', {'_a', '_x'}, {'_out'}, '_out = _a * _x')

me1.add_in_connector('IN_a');    me1.add_out_connector('OUT_a')
me1.add_in_connector('IN_x');    me1.add_out_connector('OUT_x')
mx1.add_in_connector('IN_stream'); mx1.add_out_connector('OUT_stream')

state.add_edge(a_node,       None,      me1,          'IN_a',
               dace.Memlet('a[0]'))
state.add_edge(x_node,       None,      me1,          'IN_x',
               dace.Memlet('x[0:N]'))
state.add_edge(me1,          'OUT_a',   mult_tasklet, '_a',
               dace.Memlet('a[0]'))
state.add_edge(me1,          'OUT_x',   mult_tasklet, '_x',
               dace.Memlet('x[i]'))

# Push to stream
state.add_edge(mult_tasklet, '_out',    mx1,          'IN_stream',
               dace.Memlet('tmp_stream[0]'))
state.add_edge(mx1,          'OUT_stream', stream_node, None,
               dace.Memlet('tmp_stream[0:N]'))

# ── Map 2: add — consumer, pops from stream ──────────────────────────
# CRITICAL: must be Sequential — see Section 8.4
me2, mx2 = state.add_map('add_map', {'i': '0:N'},
                         schedule=dace.ScheduleType.Sequential)
add_tasklet = state.add_tasklet(
    'add', {'_tmp', '_y'}, {'_out'}, '_out = _tmp + _y')

me2.add_in_connector('IN_stream'); me2.add_out_connector('OUT_stream')
me2.add_in_connector('IN_y');      me2.add_out_connector('OUT_y')
mx2.add_in_connector('IN_result'); mx2.add_out_connector('OUT_result')

state.add_edge(stream_node,  None,       me2,         'IN_stream',
               dace.Memlet('tmp_stream[0:N]'))
state.add_edge(y_node,       None,       me2,         'IN_y',
               dace.Memlet('y[0:N]'))
state.add_edge(me2,          'OUT_stream', add_tasklet, '_tmp',
               dace.Memlet('tmp_stream[0]'))
state.add_edge(me2,          'OUT_y',    add_tasklet,  '_y',
               dace.Memlet('y[i]'))
state.add_edge(add_tasklet,  '_out',     mx2,          'IN_result',
               dace.Memlet('result[i]'))
state.add_edge(mx2,          'OUT_result', out_node,   None,
               dace.Memlet('result[0:N]'))

sdfg.validate()
sdfg.save('axpy_stream.sdfg')

compiled = sdfg.compile()
a      = np.array([2.0])   # shape [1] — must match add_array declaration
x      = np.random.rand(1024)
y      = np.random.rand(1024)
result = np.zeros(1024)
compiled(a=a, x=x, y=y, result=result, N=1024)

expected = 2.0 * x + y
print("Max error:", np.max(np.abs(result - expected)))
```

---

### Exercise 8.3: Reading the Stream SDFG

Open `axpy_stream.sdfg` and answer:

**Q1.** What does the stream node look like visually — how is it different
from a regular AccessNode?

<details>
<summary>Expected answer</summary>

The stream node looks like an AccessNode (oval/circle) but with **dashed
borders** instead of solid. This visual distinction signals that it is a
FIFO communication channel, not a random-access memory location. You cannot
index into it arbitrarily — you can only push and pop in order.

</details>

---

**Q2.** What does `buffer_size=1` mean physically?

<details>
<summary>Expected answer</summary>

`buffer_size=1` means a **single-entry FIFO** — only one element can be
in the buffer at a time. The producer pushes one element and must wait
for the consumer to pop it before pushing the next. Producer and consumer
run in complete lock-step.

| `buffer_size` | Meaning | Hardware analogy |
|---|---|---|
| `1` | Lock-step push/pop | Single register handoff |
| `32` | Producer can run 32 iterations ahead | 32-entry shift register |
| `N` | Producer fills entirely first | Equivalent to transient array |

</details>

---

**Q3.** Inspect the stream type programmatically:

```python
import dace

sdfg = dace.SDFG.from_file('axpy_stream.sdfg')

for name, arr in sdfg.arrays.items():
    arr_type = type(arr).__name__
    print(f"{name:20s}  type={arr_type:15s}  "
          f"transient={arr.transient}")
    if isinstance(arr, dace.data.Stream):
        print(f"  buffer_size={arr.buffer_size}  "
              f"dtype={arr.dtype}")
```

What Python type does a stream have, and why is it always transient?

<details>
<summary>Expected answer</summary>

```
a                     type=Array            transient=False
x                     type=Array            transient=False
y                     type=Array            transient=False
result                type=Array            transient=False
tmp_stream            type=Stream           transient=True
  buffer_size=1  dtype=double
```

The stream has type `dace.data.Stream`. It is always `transient=True`
because a stream only exists as a communication channel between two stages
inside the SDFG — it never makes sense as an external input or output.
You cannot hand a FIFO queue to the caller; you hand them arrays.

</details>

---

## 8.4 Critical Fix: Streams Require Sequential Schedule

> **This is the most important fix in this entire part.** Getting it wrong
> produces large numerical errors that are invisible in the SDFG itself —
> they only appear at runtime.

### The bug

DaCe's default map schedule is `CPU_Multicore`, which generates
`#pragma omp parallel for` — OpenMP threads run iterations in arbitrary order.

For a FIFO stream, push and pop order must match exactly:
- Producer must push element `i` before element `i+1`
- Consumer must pop in the same order

With parallel threads:

```
Thread 3 pushes x[3] first
Thread 0 pushes x[0] second     ← scrambled order!
Consumer pops in FIFO order
→ result[i] = a*x[scrambled_j] + y[i]
→ large numerical error (~1.9 for random inputs)
```

The error is invisible in the SDFG — the memlets and connectors are all
correct. It only shows up in the generated C++ and at runtime.

### The fix

Always set `schedule=dace.ScheduleType.Sequential` on maps that communicate
through a stream:

```python
# WRONG — default CPU_Multicore causes scrambled push/pop order
me1, mx1 = state.add_map('multiply_map', {'i': '0:N'})

# CORRECT — Sequential enforces strict iteration order
me1, mx1 = state.add_map('multiply_map', {'i': '0:N'},
                         schedule=dace.ScheduleType.Sequential)
me2, mx2 = state.add_map('add_map', {'i': '0:N'},
                         schedule=dace.ScheduleType.Sequential)
```

### Why this matches hardware behavior

In real hardware, FIFOs are inherently sequential — the hardware guarantees
push/pop ordering. `dace.ScheduleType.Sequential` correctly models this
hardware behavior in DaCe's CPU simulation.

When targeting FPGA or modeling in gem5-Aladdin, the ordering is enforced
by the hardware FIFO itself. But on CPU you must set it explicitly.

### Verifying correctness

```python
# Use known values to verify — should give exactly 1.0 everywhere
a      = np.array([1.0])
x      = np.ones(1024)
y      = np.zeros(1024)
result = np.zeros(1024)

compiled(a=a, x=x, y=y, result=result, N=1024)
print("Unique values:", np.unique(result))  # should be [1.]
```

---

## 8.5 Buffer Size and Pipeline Depth

```python
import dace
import numpy as np
import time

N_val = 1024 * 1024

for buf_size in [1, 32, 1024]:
    sdfg = dace.SDFG.from_file('axpy_stream.sdfg')
    sdfg.arrays['tmp_stream'].buffer_size = buf_size
    sdfg.name = f'axpy_stream_buf{buf_size}'  # avoid recompile warning

    compiled = sdfg.compile()
    a      = np.array([2.0])
    x      = np.random.rand(N_val)
    y      = np.random.rand(N_val)
    result = np.zeros(N_val)

    # Warmup
    compiled(a=a, x=x, y=y, result=result, N=N_val)

    start = time.perf_counter()
    for _ in range(10):
        compiled(a=a, x=x, y=y, result=result, N=N_val)
    elapsed = (time.perf_counter() - start) / 10

    error = np.max(np.abs(result - (2.0 * x + y)))
    print(f"buffer_size={buf_size:6d}  "
          f"time={elapsed*1000:.3f}ms  "
          f"error={error:.2e}")
```

---

### Exercise 8.5: Buffer Size

**Q1.** Does buffer size affect timing on CPU? Why or why not?

<details>
<summary>Expected answer</summary>

No — timing is essentially identical across all buffer sizes (~55ms for
1M elements). On CPU, `dace::Stream<double>` is a software queue. Since
both maps are Sequential and run in lock-step, there is never more than
1 element in flight regardless of buffer size. The ~55ms is dominated by
two sequential passes over 1M float64 elements — pure memory bandwidth,
nothing to do with stream depth.

Expected output:
```
buffer_size=     1  time=55.100ms  error=0.00e+00
buffer_size=    32  time=54.255ms  error=0.00e+00
buffer_size=  1024  time=54.700ms  error=0.00e+00
```

</details>

---

**Q2.** Where does buffer size actually matter?

<details>
<summary>Expected answer</summary>

Buffer size matters in two real contexts:

**FPGA synthesis** — DaCe targets Xilinx/Intel FPGAs where streams map
to hardware FIFOs. `buffer_size=32` synthesizes a 32-entry shift register.
Too small → pipeline stalls. Too large → wastes FPGA LUT/BRAM resources.

**gem5-Aladdin simulation** — When modeling your CVA6 accelerator pipeline,
buffer size determines how many cycles of latency the FIFO can absorb
between stages. If your GEMM systolic array produces a result every 4
cycles but your softmax vector unit takes 6 cycles per element, you need
at least a 2-entry buffer to prevent stalls.

For your methodology, buffer size is a **hardware design parameter** in
your accelerator design space exploration — part of what gem5-Aladdin
sweeps across in Phase 4.

</details>

---

## 8.6 Streams Between Accelerator Stages

The most important use of streams for your research is modeling the
communication between your GEMM systolic array and your softmax vector unit.

```python
import dace
import numpy as np

M, N, K = dace.symbol('M'), dace.symbol('N'), dace.symbol('K')
sdfg = dace.SDFG('gemm_softmax_pipeline')

sdfg.add_array('A',      shape=[M, K], dtype=dace.float64)
sdfg.add_array('B',      shape=[K, N], dtype=dace.float64)
sdfg.add_array('result', shape=[M, N], dtype=dace.float64)

# Each stream element is a full row of N elements
# GEMM produces one row per outer iteration
# Softmax consumes one row per outer iteration
sdfg.add_stream('gemm_to_softmax',
                dtype=dace.float64,
                shape=[N],         # one row per push/pop
                buffer_size=1,
                transient=True)

state = sdfg.add_state('pipeline')

A_node      = state.add_read('A')
B_node      = state.add_read('B')
out_node    = state.add_write('result')
stream_node = state.add_access('gemm_to_softmax')

# ── Stage 1: GEMM ────────────────────────────────────────────────────
# Outer map: one iteration = compute one complete row, push it
gemm_outer_e, gemm_outer_x = state.add_map(
    'gemm_rows', {'i': '0:M'},
    schedule=dace.ScheduleType.Sequential)

# Inner map: dot product accumulation over K
gemm_inner_e, gemm_inner_x = state.add_map(
    'gemm_k', {'k': '0:K'},
    schedule=dace.ScheduleType.Sequential)

gemm_tasklet = state.add_tasklet(
    'gemm_op', {'_a', '_b'}, {'_out'}, '_out = _a * _b')

gemm_outer_e.add_in_connector('IN_A');  gemm_outer_e.add_out_connector('OUT_A')
gemm_outer_e.add_in_connector('IN_B');  gemm_outer_e.add_out_connector('OUT_B')
gemm_inner_e.add_in_connector('IN_A');  gemm_inner_e.add_out_connector('OUT_A')
gemm_inner_e.add_in_connector('IN_B');  gemm_inner_e.add_out_connector('OUT_B')
gemm_inner_x.add_in_connector('IN_stream')
gemm_inner_x.add_out_connector('OUT_stream')
gemm_outer_x.add_in_connector('IN_stream')
gemm_outer_x.add_out_connector('OUT_stream')

state.add_edge(A_node,       None,       gemm_outer_e, 'IN_A',
               dace.Memlet('A[0:M, 0:K]'))
state.add_edge(B_node,       None,       gemm_outer_e, 'IN_B',
               dace.Memlet('B[0:K, 0:N]'))
state.add_edge(gemm_outer_e, 'OUT_A',    gemm_inner_e, 'IN_A',
               dace.Memlet('A[i, 0:K]'))
state.add_edge(gemm_outer_e, 'OUT_B',    gemm_inner_e, 'IN_B',
               dace.Memlet('B[0:K, 0:N]'))
state.add_edge(gemm_inner_e, 'OUT_A',    gemm_tasklet, '_a',
               dace.Memlet('A[i, k]'))
state.add_edge(gemm_inner_e, 'OUT_B',    gemm_tasklet, '_b',
               dace.Memlet('B[k, 0:N]'))

# WCR accumulates K partial products into the stream row
state.add_edge(gemm_tasklet, '_out',     gemm_inner_x, 'IN_stream',
               dace.Memlet(data='gemm_to_softmax',
                           subset='0:N',
                           wcr='lambda x, y: x + y'))
state.add_edge(gemm_inner_x, 'OUT_stream', gemm_outer_x, 'IN_stream',
               dace.Memlet('gemm_to_softmax[0:N]'))
state.add_edge(gemm_outer_x, 'OUT_stream', stream_node, None,
               dace.Memlet('gemm_to_softmax[0:N]'))

# ── Stage 2: Softmax (placeholder) ───────────────────────────────────
# Outer map: one iteration = pop one row from stream, normalize it
sfx_outer_e, sfx_outer_x = state.add_map(
    'sfx_rows', {'i': '0:M'},
    schedule=dace.ScheduleType.Sequential)

sfx_inner_e, sfx_inner_x = state.add_map(
    'sfx_norm', {'j': '0:N'},
    schedule=dace.ScheduleType.Sequential)

sfx_tasklet = state.add_tasklet(
    'sfx_op', {'_row'}, {'_out'},
    '_out = _row'   # placeholder — replace with real softmax
)

sfx_outer_e.add_in_connector('IN_stream')
sfx_outer_e.add_out_connector('OUT_stream')
sfx_inner_e.add_in_connector('IN_stream')
sfx_inner_e.add_out_connector('OUT_stream')
sfx_inner_x.add_in_connector('IN_result')
sfx_inner_x.add_out_connector('OUT_result')
sfx_outer_x.add_in_connector('IN_result')
sfx_outer_x.add_out_connector('OUT_result')

state.add_edge(stream_node,  None,        sfx_outer_e, 'IN_stream',
               dace.Memlet('gemm_to_softmax[0:N]'))
state.add_edge(sfx_outer_e,  'OUT_stream', sfx_inner_e, 'IN_stream',
               dace.Memlet('gemm_to_softmax[0:N]'))
state.add_edge(sfx_inner_e,  'OUT_stream', sfx_tasklet, '_row',
               dace.Memlet('gemm_to_softmax[j]'))
state.add_edge(sfx_tasklet,  '_out',       sfx_inner_x, 'IN_result',
               dace.Memlet('result[i, j]'))
state.add_edge(sfx_inner_x,  'OUT_result', sfx_outer_x, 'IN_result',
               dace.Memlet('result[i, 0:N]'))
state.add_edge(sfx_outer_x,  'OUT_result', out_node,    None,
               dace.Memlet('result[0:M, 0:N]'))

sdfg.validate()
sdfg.save('gemm_softmax_pipeline.sdfg')
sdfg.view()
```

---

### Exercise 8.6: Accelerator Pipeline

**Q1.** In the SDFG viewer, trace the stream from GEMM output to softmax
input. What does the granularity of push/pop correspond to in hardware?

<details>
<summary>Expected answer</summary>

The stream sits between `gemm_rows` outer exit and `sfx_rows` outer entry.
The push/pop granularity is **one complete row of N elements** per iteration
of the outer M map. In hardware this means:

- GEMM systolic array finishes computing one row → writes row to FIFO
- Softmax vector unit reads one row from FIFO → normalizes it

The FIFO absorbs timing differences between the two stages. If GEMM
takes 256 cycles per row and softmax takes 128 cycles per row, softmax
processes two rows in the time GEMM produces one — the FIFO drains faster
than it fills, so no stalls occur.

</details>

---

**Q2.** Why can't we remove the outer M map and stream directly element by
element?

<details>
<summary>Expected answer</summary>

The outer M map serves as the **scope boundary** that defines when a
complete row is ready to push. Without it, DaCe has no point in the graph
to attach the push/pop to.

The inner K map exit is too early — at that point only one partial product
`A[i,k] * B[k,j]` is ready. The full row `C[i, 0:N]` only exists after
all K iterations complete, which corresponds to the outer M map exit.

You cannot stream at finer granularity than the accumulation scope —
the FIFO must carry complete, finalized values.

</details>

---

**Q3.** What does the stream physically represent in your CVA6
heterogeneous system?

<details>
<summary>Expected answer</summary>

In your CVA6 accelerator complex, `gemm_to_softmax` maps to an **on-chip
FIFO register file** between the systolic array output port and the vector
unit input port. Physically:

- GEMM systolic array → writes completed row to FIFO via its output bus
- Vector unit (softmax) → reads row from FIFO via its input bus
- FIFO sits entirely on-chip — no DRAM round-trip

The `buffer_size` parameter directly determines the depth of this FIFO
in RTL — a key design space exploration parameter in Phase 3 and 4 of
your methodology.

</details>

---

## 8.7 Impact on Arithmetic Intensity

Streams change the roofline equation for inter-stage communication by
eliminating DRAM traffic between pipeline stages.

**Without stream (transient array to DRAM):**

$$\text{Bytes}_{\text{total}} = \text{Bytes}_{\text{GEMM}} + \underbrace{M \times N \times 8}_{\text{write to DRAM}} + \underbrace{M \times N \times 8}_{\text{read from DRAM}} + \text{Bytes}_{\text{softmax}}$$

**With stream (on-chip FIFO):**

$$\text{Bytes}_{\text{total}} = \text{Bytes}_{\text{GEMM}} + \underbrace{0}_{\text{stream stays on-chip}} + \text{Bytes}_{\text{softmax}}$$

For $M=N=128$ (self-attention with $S=128$), the saving is:

$$\Delta\text{Bytes} = 2 \times 128 \times 128 \times 8 = 262,144 \text{ bytes} \approx 0.26 \text{ MB per attention layer}$$

This directly improves the system-level arithmetic intensity — FLOPs stay
constant but DRAM bytes decrease.

> **Note:** The arithmetic intensity extractor from Part 4 measures
> **static** bytes from array boundary memlets. It will not automatically
> account for stream savings. When streams eliminate DRAM traffic, you
> must manually adjust the bytes calculation to reflect what actually
> goes to DRAM vs what stays on-chip.

---

## 8.8 Known Limitation: CPU vs Hardware Pipeline Parallelism

On CPU, Sequential maps communicate through a software FIFO but do **not**
achieve true pipeline parallelism — they still run one after the other:

```
CPU execution:
time → [GEMM row 0][GEMM row 1][...][GEMM row M]
                                                  [SFX row 0][SFX row 1]...
```

In real hardware, pipeline stages overlap:

```
Hardware execution:
time → [GEMM row 0][GEMM row 1][GEMM row 2]...
                   [SFX row 0] [SFX row 1] ...
```

The DaCe SDFG correctly captures the **dependency structure** (GEMM row i
must complete before softmax row i starts) but actual pipelining emerges
only when targeting FPGA or modeling in gem5-Aladdin with proper pipeline
timing.

This is why Phase 4 of your methodology uses gem5-Aladdin — it models
the timing behavior that DaCe's CPU backend cannot capture.

---

## Summary: Part 8

You now understand the complete stream model in DaCe:

| Concept | Key detail |
|---|---|
| Declaration | `sdfg.add_stream(name, dtype, buffer_size, transient=True)` |
| Always transient | Streams are internal communication channels only |
| Visual indicator | Dashed oval in SDFG viewer |
| Sequential schedule | **Required** on both producer and consumer maps |
| Buffer size | FIFO depth — hardware design parameter, not CPU performance |
| Push memlet | `dace.Memlet('stream_name[0]')` on tasklet output |
| Pop memlet | `dace.Memlet('stream_name[0]')` on tasklet input |
| Granularity | Set by map scope boundary — push/pop at map exit |

**Critical fix to always remember:**

> Maps communicating via streams **must** use
> `schedule=dace.ScheduleType.Sequential`. The default `CPU_Multicore`
> schedule generates OpenMP parallel loops that push/pop out of order,
> causing large numerical errors that are invisible in the SDFG and
> only appear at runtime.

**The hardware connection:**

```
DaCe stream          →  Hardware FIFO register
buffer_size          →  FIFO depth in RTL
Sequential schedule  →  Hardware's inherent ordering guarantee
shape=[N] per push   →  Row-granularity data transfer
stream on-chip       →  Eliminates DRAM round-trip between stages
```

**For your methodology:**

Streams are the communication primitive for Phase 5 hardware design.
When you define Library Node expansions for your CVA6 accelerator
(Part 5), the accelerator expansion should use streams between pipeline
stages rather than transient arrays — this correctly models the on-chip
FIFO communication that your hardware will implement.

**Tutorial complete.** You now have the full DaCe toolkit:
Parts 1–7 gave you the analysis methodology; Part 8 gives you the
hardware modeling primitive to connect it to your accelerator design.