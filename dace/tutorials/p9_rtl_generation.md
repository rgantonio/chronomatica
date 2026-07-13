# DaCe Mastery Tutorial
## Part 9: RTL Generation with SystemVerilog Tasklets

> **Prerequisites:** Complete Parts 1–8. You should understand maps, memlets,
> Library Nodes, and streams. This part connects DaCe's analysis capabilities
> directly to hardware RTL generation.

---

## 9.1 What RTL Generation Means in DaCe

DaCe's SystemVerilog tasklet feature does **not** automatically synthesize
your dataflow graph into RTL. Instead it gives you a clean division of
responsibility:

```
You write:    the compute logic in SystemVerilog (inside the tasklet)
DaCe writes:  the AXI-Stream interface wrapper, port declarations,
              clock/reset signals, and integration glue
```

The result is a complete, synthesizable SystemVerilog module that can
be fed directly into your synthesis flow (Genus, Design Compiler, or
your SNAX shell generator).

This is the right abstraction for your CVA6 heterogeneous system —
you understand the compute kernels deeply, and DaCe handles the boilerplate.

---

## 9.2 Two Paths from DaCe to RTL

| Path | Flow | Use when |
|---|---|---|
| **HLS** | DaCe → HLS C++ → Vivado/Vitis → Verilog | Targeting commercial FPGAs |
| **Direct RTL** | DaCe → SystemVerilog tasklet → `.sv` | Targeting custom SoC (CVA6/SNAX) |

For your SNAX/CVA6 system, **Path 2 is the right choice**. It produces
SystemVerilog directly without requiring any Xilinx or Intel vendor toolkit.

---

## 9.3 Two Important API Fixes

Before any code, two fixes discovered during development:

**Fix 1 — Specialize symbols before RTL codegen**

The RTL backend must unroll maps at codegen time, so all map bounds
must be concrete integers. Free symbols like `N` cause codegen failure:

```python
# Required before generate_code() for RTL tasklets
sdfg.specialize({'N': 4})
```

This is not a workaround — it reflects the fundamental nature of RTL.
Every wire and port must have a fixed bit width at synthesis time.
In your SNAX methodology this maps to **parameter elaboration** — you
pick concrete dimensions before generating RTL.

**Fix 2 — Use `generate_program_folder` to write files to disk**

`sdfg.generate_code()` returns code objects in memory but does not
write them to disk. `sdfg.compile()` writes files but also triggers
a full CMake/Verilator build. To generate without building:

```python
from dace.codegen.compiler import generate_program_folder

program_code = sdfg.generate_code()
generate_program_folder(sdfg, program_code,
                        os.path.join('.dacecache', sdfg.name))
```

---

## 9.4 The Generated AXI-Stream Interface

DaCe wraps every SystemVerilog tasklet with an **AXI-Stream interface**
using Xilinx AP (Accelerator Protocol) conventions. Understanding every
signal is essential before writing tasklet code.

### Control signals (`ap_` prefix)

```systemverilog
input  ap_aclk    // clock — all registers trigger on posedge ap_aclk
input  ap_areset  // reset — active high synchronous reset
input  ap_start   // host → kernel: "begin processing"
output ap_done    // kernel → host: "I have finished"
```

`ap_` stands for **Accelerator Protocol**. In your CVA6 system:
- `ap_start` comes from a CSR write by the host CPU
- `ap_done` triggers an interrupt or status register update

### Input streams (`s_axis_` prefix — Slave)

```systemverilog
input  s_axis_a_tvalid  // producer: "my data is valid"
input  s_axis_a_tdata   // the data — 32 bits for float32
output s_axis_a_tready  // consumer: "I am ready to accept"
input  s_axis_a_tkeep   // byte enables — which bytes are valid
input  s_axis_a_tlast   // end of burst marker
```

`s_axis` = **Slave AXI-Stream** — the module is the receiver.
Your input arrays (`a`, `b`) arrive as AXI-Stream inputs.

### Output streams (`m_axis_` prefix — Master)

```systemverilog
output m_axis_c_tvalid  // "I have valid output data"
output m_axis_c_tdata   // the result — 32 bits
input  m_axis_c_tready  // downstream: "I am ready to accept"
output m_axis_c_tkeep   // byte enables
output m_axis_c_tlast   // end of burst
```

`m_axis` = **Master AXI-Stream** — the module drives the output.
Your result array (`c`) leaves as an AXI-Stream output.

### The AXI-Stream handshake rule

```
Data transfers when: tvalid AND tready are BOTH high on the same clock edge
```

This is the only rule you need to implement correctly in your tasklet code.

### Port width for `float32`

`float32` = 32 bits = 4 bytes → `tdata` is 32 bits, `tkeep` is 4 bits
(one bit per byte). `tkeep=4'hF` means all four bytes are valid.

---

## 9.5 Combinational MAC — Your First RTL Tasklet

```python
import dace
import numpy as np
import os
from dace.codegen.compiler import generate_program_folder

N = dace.symbol('N')
sdfg = dace.SDFG('rtl_mac')

sdfg.add_array('a', shape=[N], dtype=dace.float32, transient=False)
sdfg.add_array('b', shape=[N], dtype=dace.float32, transient=False)
sdfg.add_array('c', shape=[N], dtype=dace.float32, transient=False)

state = sdfg.add_state('compute')

a_node = state.add_read('a')
b_node = state.add_read('b')
c_node = state.add_write('c')

me, mx = state.add_map('mac_map', {'i': '0:N'},
                       schedule=dace.ScheduleType.Sequential)

tasklet = state.add_tasklet(
    name='mac_rtl',
    inputs={'a', 'b'},
    outputs={'c'},
    code='''
        // Combinational multiply using AXI-Stream port names
        // Accept inputs when downstream is ready
        assign s_axis_a_tready = m_axis_c_tready;
        assign s_axis_b_tready = m_axis_c_tready;

        // Compute result combinationally
        assign m_axis_c_tdata  = s_axis_a_tdata * s_axis_b_tdata;

        // Output valid when both inputs valid
        assign m_axis_c_tvalid = s_axis_a_tvalid & s_axis_b_tvalid;
        assign m_axis_c_tkeep  = 4'hF;
        assign m_axis_c_tlast  = s_axis_a_tlast;

        // Done when last element transferred
        assign ap_done = m_axis_c_tvalid & m_axis_c_tready
                         & m_axis_c_tlast;
    ''',
    language=dace.dtypes.Language.SystemVerilog
)

me.add_in_connector('IN_a');  me.add_out_connector('OUT_a')
me.add_in_connector('IN_b');  me.add_out_connector('OUT_b')
mx.add_in_connector('IN_c');  mx.add_out_connector('OUT_c')

state.add_edge(a_node, None, me,       'IN_a', dace.Memlet('a[0:N]'))
state.add_edge(b_node, None, me,       'IN_b', dace.Memlet('b[0:N]'))
state.add_edge(me,  'OUT_a', tasklet,  'a',    dace.Memlet('a[i]'))
state.add_edge(me,  'OUT_b', tasklet,  'b',    dace.Memlet('b[i]'))
state.add_edge(tasklet, 'c', mx,    'IN_c',    dace.Memlet('c[i]'))
state.add_edge(mx, 'OUT_c', c_node,    None,   dace.Memlet('c[0:N]'))

sdfg.validate()
sdfg.save('rtl_mac.sdfg')

# Fix 1: specialize before RTL codegen
sdfg.specialize({'N': 4})

# Fix 2: generate without building
program_code = sdfg.generate_code()
generate_program_folder(sdfg, program_code,
                        os.path.join('.dacecache', sdfg.name))

# Find and print generated RTL
import glob
sv_files = glob.glob('.dacecache/**/*.sv', recursive=True)
v_files  = glob.glob('.dacecache/**/*.v',  recursive=True)

print(f"Found {len(sv_files + v_files)} RTL files:")
for f in sv_files + v_files:
    print(f"\n{'='*60}")
    print(f"File: {f}")
    print('='*60)
    with open(f, 'r') as fh:
        print(fh.read())
```

---

### Exercise 9.5: Reading the Generated Module

**Q1.** Find your `assign m_axis_c_tdata = s_axis_a_tdata * s_axis_b_tdata`
in the generated `.sv`. What module name did DaCe give it?

<details>
<summary>Expected answer</summary>

DaCe names the module after the tasklet label plus indices:
`mac_rtl_0_0_5` (or similar). The number suffix encodes the SDFG,
state, and node indices. Your assign statement appears inside the
module body exactly as written.

</details>

---

**Q2.** What is the `tkeep` port width and why?

<details>
<summary>Expected answer</summary>

`tkeep` is 4 bits because `float32` = 32 bits = 4 bytes, and `tkeep`
has one bit per byte. `tkeep=4'hF` (all ones) means all four bytes
of every transfer are valid. For `float64` you would see `tkeep` as
8 bits.

</details>

---

**Q3.** Why must you use `s_axis_a_tdata` in your tasklet code rather
than just `a`?

<details>
<summary>Expected answer</summary>

DaCe declares the data ports as `s_axis_a_tdata` and `m_axis_c_tdata`
— these are the actual wire names in the generated module. The tasklet
input connector `a` maps to `s_axis_a_tdata` in the interface, but
inside the module body you must use the full port name. Using bare `a`
would reference an undeclared wire and fail synthesis.

</details>

---

## 9.6 Pipelined MAC — Registered Pipeline Stages

Combinational logic is limited by timing closure. Real accelerators
use registered pipelines to achieve high clock frequencies.

```python
tasklet = state.add_tasklet(
    name='mac_pipelined',
    inputs={'a', 'b'},
    outputs={'c'},
    code='''
        // 2-stage registered pipeline
        logic [31:0] stage1_data;
        logic        stage1_valid;

        // Pipeline enable — only advance when downstream is ready
        wire pipe_en = m_axis_c_tready;

        // Accept inputs when pipeline can advance
        assign s_axis_a_tready = pipe_en;
        assign s_axis_b_tready = pipe_en;

        always_ff @(posedge ap_aclk) begin
            if (ap_areset) begin
                stage1_data     <= 32'b0;
                stage1_valid    <= 1'b0;
                m_axis_c_tdata  <= 32'b0;
                m_axis_c_tvalid <= 1'b0;
                m_axis_c_tlast  <= 1'b0;
            end else if (pipe_en) begin
                // Stage 1: multiply
                stage1_data  <= s_axis_a_tdata * s_axis_b_tdata;
                stage1_valid <= s_axis_a_tvalid & s_axis_b_tvalid;

                // Stage 2: register output
                m_axis_c_tdata  <= stage1_data;
                m_axis_c_tvalid <= stage1_valid;
                m_axis_c_tlast  <= s_axis_a_tlast;
            end
        end

        assign m_axis_c_tkeep = 4'hF;
        assign ap_done = m_axis_c_tvalid & m_axis_c_tready
                         & m_axis_c_tlast;
    ''',
    language=dace.dtypes.Language.SystemVerilog
)
```

---

### Exercise 9.6: Pipeline Analysis

**Q1.** How many cycles does data take from input to output?

<details>
<summary>Expected answer</summary>

**2 cycles:**
- Cycle 1: `s_axis_a_tdata * s_axis_b_tdata` → `stage1_data`
- Cycle 2: `stage1_data` → `m_axis_c_tdata`

This means the first valid output appears 2 cycles after the first
valid input. For your SNAX shell configuration:
`num_pipeline_stages = 2`, `latency = 2 cycles`.

</details>

---

**Q2.** What is the initiation interval — how often can new data enter
the pipeline?

<details>
<summary>Expected answer</summary>

**1 cycle** — every cycle when `pipe_en` is high, a new element enters
stage 1. The pipeline is fully pipelined: throughput = 1 result per
cycle (after the initial 2-cycle latency). This maps to
`initiation_interval = 1` in your SNAX shell parameters.

</details>

---

**Q3.** What happens when `m_axis_c_tready` goes low (backpressure)?

<details>
<summary>Expected answer</summary>

`pipe_en` goes low, freezing the entire pipeline — no new data enters
stage 1, no data advances from stage 1 to stage 2. This is **pipeline
stalling** and is the correct AXI-Stream compliant behavior.

In your CVA6 system, backpressure comes from the TCDM (Tightly Coupled
Data Memory). If the scratchpad cannot accept a result, it asserts
backpressure and your accelerator stalls until the path clears.

</details>

---

## 9.7 Deriving SNAX Shell Parameters Automatically

The most powerful application of this for your research — automatically
derive SNAX shell parameters from a DaCe SDFG with RTL tasklets:

```python
import dace
import re
import sympy

def derive_snax_params(sdfg_path, symbol_values):
    """
    Derive SNAX shell parameters from a DaCe SDFG containing
    RTL tasklets. Combines roofline analysis with RTL interface
    derivation.

    Returns dict of SNAX shell configuration parameters.
    """
    sdfg = dace.SDFG.from_file(sdfg_path)
    params = {}

    for state in sdfg.states():
        for node in state.nodes():

            # RTL tasklet — derive interface parameters
            if isinstance(node, dace.nodes.Tasklet):
                if node.language == dace.dtypes.Language.SystemVerilog:
                    print(f"RTL Tasklet: {node.label}")

                    # Data width from connected array dtype
                    for edge in state.in_edges(node):
                        if (edge.data.data is not None
                                and edge.data.data in sdfg.arrays):
                            arr = sdfg.arrays[edge.data.data]
                            params['data_width'] = arr.dtype.bytes * 8
                            break

                    # Port counts
                    params['num_input_ports']  = \
                        len(node.in_connectors)
                    params['num_output_ports'] = \
                        len(node.out_connectors)

                    # Pipeline depth from always_ff count
                    code = node.code.as_string
                    ff_count = len(re.findall(r'always_ff', code))
                    params['num_pipeline_stages'] = max(ff_count, 1)

                    # Fully pipelined assumption
                    params['initiation_interval'] = 1

                    # AXI-Stream interface (DaCe default)
                    params['interface'] = 'AXI-Stream'
                    params['clock']     = 'ap_aclk'
                    params['reset']     = 'ap_areset (active-high)'

            # Map range — derive problem size and parallelism
            if isinstance(node, dace.nodes.MapEntry):
                ranges = {}
                for i, r in enumerate(node.map.range):
                    start, stop, step = r
                    count = sympy.simplify(
                        (stop - start + 1) / step)
                    try:
                        ranges[f'dim_{i}'] = int(
                            count.subs(symbol_values))
                    except Exception:
                        ranges[f'dim_{i}'] = str(count)
                params['problem_size'] = ranges

    # Print derived parameters
    print(f"\n{'='*50}")
    print("SNAX Shell Parameters")
    print(f"{'='*50}")
    for k, v in params.items():
        print(f"  {k:<25} {v}")
    print(f"{'='*50}")

    return params


# Test on pipelined MAC SDFG
params = derive_snax_params('rtl_mac.sdfg', {'N': 4})
```

Expected output:

```
RTL Tasklet: mac_pipelined
==================================================
SNAX Shell Parameters
==================================================
  data_width                32
  num_input_ports           2
  num_output_ports          1
  num_pipeline_stages       2
  initiation_interval       1
  interface                 AXI-Stream
  clock                     ap_aclk
  reset                     ap_areset (active-high)
  problem_size              {'dim_0': 4}
==================================================
```

---

### Exercise 9.7: SNAX Parameter Derivation

**Q1.** How does `num_pipeline_stages = 2` map to your SNAX shell
generator?

<details>
<summary>Expected answer</summary>

The SNAX shell generator uses `num_pipeline_stages` to:
- Set the depth of the valid/done pipeline in the CSR controller
- Determine how many cycles after `ap_start` to expect `ap_done`
- Size any internal buffering between the accelerator and TCDM

With `num_pipeline_stages = 2` and `initiation_interval = 1`, the
shell knows the accelerator produces one result per cycle with a
fixed 2-cycle latency — it can pipeline TCDM writes accordingly.

</details>

---

**Q2.** The derived `interface = 'AXI-Stream'` but SNAX uses a TCDM
interface. What is needed to bridge them?

<details>
<summary>Expected answer</summary>

A thin **adapter module** that converts between AXI-Stream and TCDM:

```
AXI-Stream              TCDM
─────────────           ────────────
tvalid/tready    →      req/gnt
tdata            →      data
tlast            →      last_word flag
```

This adapter is exactly what your SNAX shell generator produces — making
DaCe's RTL output a natural input to your existing flow. The shell
generator wraps your accelerator (which speaks AXI-Stream) in a TCDM
compliant interface for the CVA6 bus.

</details>

---

## 9.8 The Complete DaCe → SNAX Pipeline

Combining everything from Parts 1–9, your complete research pipeline is:

```
Phase 1: Workload Characterization (Parts 1-4)
─────────────────────────────────────────────────────────────
@dace.program → SDFG → expand_library_nodes()
inventory_sdfg()          ← understand structure
collect_roofline_data()   ← FLOPs + bytes + intensity
Timer instrumentation     ← wall-clock timing
         ↓
per-component roofline table

Phase 2: Bottleneck Identification (Part 4, 7)
─────────────────────────────────────────────────────────────
I >> ridge → compute-bound → systolic array
I << ridge → memory-bound  → vector unit
I ≈ 0      → data movement → DMA / stream
irregular  → CPU only      → NeSy symbolic
         ↓
hardware dispatch map

Phase 3: Design Space Exploration (Part 5)
─────────────────────────────────────────────────────────────
Library Node per bottleneck kernel
    ├── pure expansion      → correctness validation
    ├── cpu_blas expansion  → CPU baseline
    └── accelerator expansion → model your design
swap implementations → profile → compare
         ↓
candidate accelerator configurations

Phase 4: System Modeling
─────────────────────────────────────────────────────────────
gem5-Aladdin → SystemC TLM → CVA6 Verilator
streams (Part 8) model inter-stage FIFOs
buffer_size = hardware FIFO depth parameter
         ↓
cycle-accurate performance validation

Phase 5: RTL Generation (Part 9)
─────────────────────────────────────────────────────────────
sdfg.specialize({'N': concrete_value})
SystemVerilog tasklet → generate_program_folder()
derive_snax_params()  → SNAX shell parameters
         ↓
SNAX shell generator → SystemVerilog modules
         ↓
CVA6 integration → FPGA prototype → ASIC
```

---

## Summary: Part 9

You now have a complete path from SDFG analysis to synthesizable RTL:

| Concept | Key detail |
|---|---|
| `language=SystemVerilog` | Tasklet code emitted as RTL, not C++ |
| `sdfg.specialize({'N': val})` | Required — RTL needs concrete port widths |
| `generate_program_folder()` | Writes `.sv` to disk without building |
| `ap_aclk` / `ap_areset` | Clock and reset — Xilinx AP convention |
| `s_axis_*` | Input AXI-Stream ports (Slave) |
| `m_axis_*` | Output AXI-Stream ports (Master) |
| `tvalid && tready` | The handshake rule — data transfers when both high |
| `pipe_en = m_axis_c_tready` | Backpressure stalls entire pipeline |
| `always_ff` count | Directly gives `num_pipeline_stages` for SNAX |
| AXI-Stream → TCDM | Bridged by SNAX shell generator |

**The division of responsibility:**

```
You write:    compute logic in SystemVerilog inside tasklets
DaCe writes:  AXI-Stream wrapper, port declarations, clock/reset
SNAX shell:   TCDM adapter, CSR controller, CVA6 integration
```

**Key insight for your thesis:**

The DaCe SDFG is the unifying representation across all five phases.
The same graph that you profile in Phase 1 generates the RTL parameters
in Phase 5 — closing the loop between workload characterization and
hardware implementation. This closed-loop methodology, using DaCe as
the central backbone, is your core research contribution.