import dace
import numpy as np

N = dace.symbol('N')

@dace.program
def axpy(a: dace.float64, x: dace.float64[N], y: dace.float64[N]):
    return a * x + y

sdfg = axpy.to_sdfg()
sdfg.simplify()          # clean up redundant states
sdfg.save('axpy.sdfg')   # open in VS Code SDFG viewer
# sdfg.view()            # (optional) also opens in browser