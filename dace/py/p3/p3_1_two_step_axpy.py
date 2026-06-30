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

# Applying fusion
from dace.transformation.dataflow import MapFusion

sdfg.apply_transformations(MapFusion)
sdfg.save('axpy_twostep_after.sdfg')