import dace
import numpy as np

N = dace.symbol('N')
sdfg = dace.SDFG('conditional_manual')

sdfg.add_array('x',      shape=[N], dtype=dace.float64)
sdfg.add_array('result', shape=[N], dtype=dace.float64)
sdfg.add_scalar('flag', dtype=dace.int32, transient=False)

# ── States ──────────────────────────────────────────────────────────
guard  = sdfg.add_state('guard')      # no computation — pure branching
s_pos  = sdfg.add_state('scale_pos')  # flag > 0 branch
s_neg  = sdfg.add_state('scale_neg')  # flag <= 0 branch
merge  = sdfg.add_state('merge')      # rejoin — no computation

# ── Interstate edges ─────────────────────────────────────────────────
sdfg.add_edge(guard, s_pos,
              dace.InterstateEdge(condition='flag > 0'))
sdfg.add_edge(guard, s_neg,
              dace.InterstateEdge(condition='flag <= 0'))
sdfg.add_edge(s_pos, merge,
              dace.InterstateEdge())   # unconditional
sdfg.add_edge(s_neg, merge,
              dace.InterstateEdge())   # unconditional

# ── s_pos: result[i] = x[i] * 2.0 ──────────────────────────────────
x_node_pos      = s_pos.add_read('x')
result_node_pos = s_pos.add_write('result')
me_pos, mx_pos  = s_pos.add_map('map_pos', {'i': '0:N'})

me_pos.add_in_connector('IN_x');   me_pos.add_out_connector('OUT_x')
mx_pos.add_in_connector('IN_result'); mx_pos.add_out_connector('OUT_result')

tasklet_pos = s_pos.add_tasklet('scale_pos', {'_x'}, {'_result'},
                                '_result = _x * 2.0')

s_pos.add_edge(x_node_pos,    None,      me_pos,      'IN_x',
               dace.Memlet('x[0:N]'))
s_pos.add_edge(me_pos,        'OUT_x',   tasklet_pos, '_x',
               dace.Memlet('x[i]'))
s_pos.add_edge(tasklet_pos,   '_result', mx_pos,      'IN_result',
               dace.Memlet('result[i]'))
s_pos.add_edge(mx_pos,        'OUT_result', result_node_pos, None,
               dace.Memlet('result[0:N]'))

# ── s_neg: result[i] = x[i] * 0.5 ──────────────────────────────────
x_node_neg      = s_neg.add_read('x')
result_node_neg = s_neg.add_write('result')
me_neg, mx_neg  = s_neg.add_map('map_neg', {'i': '0:N'})

me_neg.add_in_connector('IN_x');   me_neg.add_out_connector('OUT_x')
mx_neg.add_in_connector('IN_result'); mx_neg.add_out_connector('OUT_result')

tasklet_neg = s_neg.add_tasklet('scale_neg', {'_x'}, {'_result'},
                                '_result = _x * 0.5')

s_neg.add_edge(x_node_neg,    None,      me_neg,      'IN_x',
               dace.Memlet('x[0:N]'))
s_neg.add_edge(me_neg,        'OUT_x',   tasklet_neg, '_x',
               dace.Memlet('x[i]'))
s_neg.add_edge(tasklet_neg,   '_result', mx_neg,      'IN_result',
               dace.Memlet('result[i]'))
s_neg.add_edge(mx_neg,        'OUT_result', result_node_neg, None,
               dace.Memlet('result[0:N]'))

sdfg.validate()
sdfg.save('conditional_manual.sdfg')

# Test both branches
compiled = sdfg.compile()
x = np.random.rand(1024)

result_pos = np.zeros(1024)
compiled(x=x, result=result_pos, flag=np.int32(1), N=1024)
print("flag=1  max error:", np.max(np.abs(result_pos - x * 2.0)))

result_neg = np.zeros(1024)
compiled(x=x, result=result_neg, flag=np.int32(-1), N=1024)
print("flag=-1 max error:", np.max(np.abs(result_neg - x * 0.5)))