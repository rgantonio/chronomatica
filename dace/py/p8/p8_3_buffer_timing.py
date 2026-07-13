import dace
import numpy as np
import time

N_val = 1024 * 1024

for buf_size in [1, 32, 1024]:
    sdfg = dace.SDFG.from_file('axpy_stream.sdfg')

    # Update buffer size
    sdfg.arrays['tmp_stream'].buffer_size = buf_size

    compiled = sdfg.compile()
    a = np.array([2.0])
    x = np.random.rand(N_val)
    y = np.random.rand(N_val)
    result = np.zeros(N_val)

    start = time.perf_counter()
    for _ in range(10):
        compiled(a=a, x=x, y=y, result=result, N=N_val)
    elapsed = (time.perf_counter() - start) / 10

    print(f"buffer_size={buf_size:6d}  "
          f"time={elapsed*1000:.3f}ms")