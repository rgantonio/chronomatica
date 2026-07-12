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