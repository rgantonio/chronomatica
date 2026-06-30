import dace
import numpy as np
import glob
from dace.transformation.dataflow import MapFusion
from dace.codegen.instrumentation.report import InstrumentationReport

N = dace.symbol('N')

@dace.program
def axpy_twostep(a: dace.float64,
                 x: dace.float64[N],
                 y: dace.float64[N]):
    tmp = a * x
    return tmp + y

# ── Before fusion ────────────────────────────────────────────────────

sdfg_before = axpy_twostep.to_sdfg()
sdfg_before.simplify()

# Instrument all maps
for state in sdfg_before.states():
    for node in state.nodes():
        if isinstance(node, dace.nodes.MapEntry):
            node.map.instrument = dace.InstrumentationType.Timer

sdfg_before.save('axpy_twostep_before.sdfg')
compiled_before = sdfg_before.compile()

# Run
N_val = 1024 * 1024
a     = np.float64(2.0)
x     = np.random.rand(N_val)
y     = np.random.rand(N_val)
out   = np.zeros(N_val)

for _ in range(10):
    compiled_before(a=a, x=x, y=y, __return=out, N=N_val)

# Read report
report_files = sorted(glob.glob(
    '.dacecache/**/report-*.json', recursive=True))
report_before = InstrumentationReport(report_files[-1])
print("=== BEFORE FUSION ===")
print(report_before)

# ── After fusion ─────────────────────────────────────────────────────

sdfg_after = axpy_twostep.to_sdfg()
sdfg_after.simplify()
sdfg_after.apply_transformations(MapFusion)

# Instrument all maps
for state in sdfg_after.states():
    for node in state.nodes():
        if isinstance(node, dace.nodes.MapEntry):
            node.map.instrument = dace.InstrumentationType.Timer

sdfg_after.save('axpy_twostep_after.sdfg')
compiled_after = sdfg_after.compile()

for _ in range(10):
    compiled_after(a=a, x=x, y=y, __return=out, N=N_val)

report_files = sorted(glob.glob(
    '.dacecache/**/report-*.json', recursive=True))
report_after = InstrumentationReport(report_files[-1])
print("=== AFTER FUSION ===")
print(report_after)

# ── Summary ───────────────────────────────────────────────────────────
print("\n=== ROOFLINE ANALYSIS ===")

# AXPY FLOPs: 2 ops per element (multiply + add)
flops = 2 * N_val

# Bytes before fusion: 1 + 5N (a + x + tmp_write + tmp_read + y + return)
bytes_before = (1 + 5 * N_val) * 8
I_before = flops / bytes_before

# Bytes after fusion: 1 + 3N (a + x + y + return)
bytes_after = (1 + 3 * N_val) * 8
I_after = flops / bytes_after

print(f"FLOPs:           {flops:,}")
print(f"Bytes (before):  {bytes_before/1e6:.2f} MB  "
      f"I = {I_before:.4f} FLOPs/byte")
print(f"Bytes (after):   {bytes_after/1e6:.2f} MB  "
      f"I = {I_after:.4f} FLOPs/byte")
print(f"Intensity ratio: {I_after/I_before:.2f}x improvement")