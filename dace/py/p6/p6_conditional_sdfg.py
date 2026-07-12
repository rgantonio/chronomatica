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