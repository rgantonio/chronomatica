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