import dace
import numpy as np

M, N, K = dace.symbol('M'), dace.symbol('N'), dace.symbol('K')

@dace.program
def matmul(A: dace.float64[M, K], B: dace.float64[K, N]):
    return A @ B

sdfg = matmul.to_sdfg()
sdfg.simplify()
sdfg.save('matmul_auto.sdfg')

sdfg.expand_library_nodes()   # ← expand before saving
sdfg.save('matmul_auto_expand.sdfg')
