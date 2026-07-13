import dace
import numpy as np

M, N, K = dace.symbol('M'), dace.symbol('N'), dace.symbol('K')
sdfg = dace.SDFG('gemm_softmax_stream')

sdfg.add_array('A',      shape=[M, K], dtype=dace.float64)
sdfg.add_array('B',      shape=[K, N], dtype=dace.float64)
sdfg.add_array('result', shape=[M, N], dtype=dace.float64)

# Each stream element is a full row of size N
sdfg.add_stream('row_stream',
                dtype=dace.float64,
                shape=[N],
                buffer_size=1,
                transient=True)

state = sdfg.add_state('pipeline')

A_node      = state.add_read('A')
B_node      = state.add_read('B')
out_node    = state.add_write('result')
stream_node = state.add_access('row_stream')

# ── Stage 1: GEMM — one outer iteration = one complete row ──────────
# Outer map over M rows — Sequential because stream order matters
gemm_outer_e, gemm_outer_x = state.add_map(
    'gemm_rows', {'i': '0:M'},
    schedule=dace.ScheduleType.Sequential)

# Inner map over K — computes dot product, accumulates into row
gemm_inner_e, gemm_inner_x = state.add_map(
    'gemm_k', {'k': '0:K'},
    schedule=dace.ScheduleType.Sequential)

gemm_tasklet = state.add_tasklet(
    'gemm_op', {'_a', '_b'}, {'_out'},
    '_out = _a * _b')

# Connectors
gemm_outer_e.add_in_connector('IN_A')
gemm_outer_e.add_in_connector('IN_B')
gemm_outer_e.add_out_connector('OUT_A')
gemm_outer_e.add_out_connector('OUT_B')

gemm_inner_e.add_in_connector('IN_A')
gemm_inner_e.add_in_connector('IN_B')
gemm_inner_e.add_out_connector('OUT_A')
gemm_inner_e.add_out_connector('OUT_B')

gemm_inner_x.add_in_connector('IN_stream')
gemm_inner_x.add_out_connector('OUT_stream')

# Push directly from inner exit to stream — no intermediate outer exit
gemm_outer_x.add_in_connector('IN_stream')
gemm_outer_x.add_out_connector('OUT_stream')

# GEMM input edges
state.add_edge(A_node,        None,      gemm_outer_e, 'IN_A',
               dace.Memlet('A[0:M, 0:K]'))
state.add_edge(B_node,        None,      gemm_outer_e, 'IN_B',
               dace.Memlet('B[0:K, 0:N]'))
state.add_edge(gemm_outer_e,  'OUT_A',   gemm_inner_e, 'IN_A',
               dace.Memlet('A[i, 0:K]'))
state.add_edge(gemm_outer_e,  'OUT_B',   gemm_inner_e, 'IN_B',
               dace.Memlet('B[0:K, 0:N]'))
state.add_edge(gemm_inner_e,  'OUT_A',   gemm_tasklet, '_a',
               dace.Memlet('A[i, k]'))
state.add_edge(gemm_inner_e,  'OUT_B',   gemm_tasklet, '_b',
               dace.Memlet('B[k, 0:N]'))

# Push row to stream — WCR accumulates K partial products
state.add_edge(gemm_tasklet,  '_out',    gemm_inner_x, 'IN_stream',
               dace.Memlet(data='row_stream',
                           subset='0:N',
                           wcr='lambda x, y: x + y'))
state.add_edge(gemm_inner_x,  'OUT_stream', gemm_outer_x, 'IN_stream',
               dace.Memlet('row_stream[0:N]'))
state.add_edge(gemm_outer_x,  'OUT_stream', stream_node,  None,
               dace.Memlet('row_stream[0:N]'))

# ── Stage 2: Softmax — pops one row per outer iteration ─────────────
# Outer map over M — Sequential to match stream order
sfx_outer_e, sfx_outer_x = state.add_map(
    'sfx_rows', {'i': '0:M'},
    schedule=dace.ScheduleType.Sequential)

# Inner map over N — normalize each element
sfx_inner_e, sfx_inner_x = state.add_map(
    'sfx_norm', {'j': '0:N'},
    schedule=dace.ScheduleType.Sequential)

sfx_tasklet = state.add_tasklet(
    'sfx_op', {'_row'}, {'_out'},
    '_out = _row'   # placeholder — replace with real softmax
)

sfx_outer_e.add_in_connector('IN_stream')
sfx_outer_e.add_out_connector('OUT_stream')
sfx_inner_e.add_in_connector('IN_stream')
sfx_inner_e.add_out_connector('OUT_stream')
sfx_inner_x.add_in_connector('IN_result')
sfx_inner_x.add_out_connector('OUT_result')
sfx_outer_x.add_in_connector('IN_result')
sfx_outer_x.add_out_connector('OUT_result')

# Softmax edges — pops one row from stream per outer iteration
state.add_edge(stream_node,  None,       sfx_outer_e,  'IN_stream',
               dace.Memlet('row_stream[0:N]'))
state.add_edge(sfx_outer_e,  'OUT_stream', sfx_inner_e, 'IN_stream',
               dace.Memlet('row_stream[0:N]'))
state.add_edge(sfx_inner_e,  'OUT_stream', sfx_tasklet, '_row',
               dace.Memlet('row_stream[j]'))
state.add_edge(sfx_tasklet,  '_out',     sfx_inner_x,  'IN_result',
               dace.Memlet('result[i, j]'))
state.add_edge(sfx_inner_x,  'OUT_result', sfx_outer_x, 'IN_result',
               dace.Memlet('result[i, 0:N]'))
state.add_edge(sfx_outer_x,  'OUT_result', out_node,    None,
               dace.Memlet('result[0:M, 0:N]'))

sdfg.validate()
sdfg.save('gemm_softmax_clean.sdfg')
sdfg.view()