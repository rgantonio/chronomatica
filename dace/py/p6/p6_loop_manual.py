import dace
import numpy as np

N = dace.symbol('N')
sdfg = dace.SDFG('loop_sum_manual')

sdfg.add_array('x',      shape=[N], dtype=dace.float64)
sdfg.add_array('result', shape=[1], dtype=dace.float64)
sdfg.add_symbol('i', dace.int32)  # loop counter as interstate variable

# ── States ──────────────────────────────────────────────────────────
init  = sdfg.add_state('init')
guard = sdfg.add_state('loop_guard')
body  = sdfg.add_state('loop_body')
after = sdfg.add_state('after_loop')

# ── Interstate edges ─────────────────────────────────────────────────
sdfg.add_edge(init,  guard,
              dace.InterstateEdge(assignments={'i': '0'}))
sdfg.add_edge(guard, body,
              dace.InterstateEdge(condition='i < N'))
sdfg.add_edge(guard, after,
              dace.InterstateEdge(condition='i >= N'))
sdfg.add_edge(body,  guard,                          # ← back-edge
              dace.InterstateEdge(assignments={'i': 'i + 1'}))

# ── init: result[0] = 0 ─────────────────────────────────────────────
result_init    = init.add_write('result')
init_tasklet   = init.add_tasklet('init_result', {}, {'_out'},
                                  '_out = 0')
init.add_edge(init_tasklet, '_out', result_init, None,
              dace.Memlet('result[0]'))

# ── loop body: result[0] += x[i] ────────────────────────────────────
# result is both read AND written — needs two separate AccessNodes
x_read      = body.add_read('x')
result_read  = body.add_read('result')    # reads old value
result_write = body.add_write('result')   # writes new value

body_tasklet = body.add_tasklet(
    'update_result',
    {'_x', '_result_in'},
    {'_result_out'},
    '_result_out = _result_in + _x'
)

body.add_edge(x_read,       None,          body_tasklet, '_x',
              dace.Memlet('x[i]'))
body.add_edge(result_read,  None,          body_tasklet, '_result_in',
              dace.Memlet('result[0]'))
body.add_edge(body_tasklet, '_result_out', result_write, None,
              dace.Memlet('result[0]'))

sdfg.validate()
sdfg.save('loop_sum_manual.sdfg')

compiled = sdfg.compile()
x = np.random.rand(16)
result = np.zeros(1)
compiled(x=x, result=result, N=16)
print("Result:  ", result[0])
print("Expected:", np.sum(x))
print("Error:   ", abs(result[0] - np.sum(x)))