import dace
import numpy as np

M, N, K = dace.symbol('M'), dace.symbol('N'), dace.symbol('K')
sdfg = dace.SDFG('matmul_manual')

# Arrays
sdfg.add_array('A', shape=[M, K], dtype=dace.float64)
sdfg.add_array('B', shape=[K, N], dtype=dace.float64)
sdfg.add_array('C', shape=[M, N], dtype=dace.float64)

state = sdfg.add_state('compute')

A_node = state.add_read('A')
B_node = state.add_read('B')
C_node = state.add_write('C')

outer_entry, outer_exit = state.add_map('outer', {'i': '0:M', 'j': '0:N'})
inner_entry, inner_exit = state.add_map('inner', {'k': '0:K'})

mac_tasklet = state.add_tasklet(
    name='mac',
    inputs={'_a', '_b'},
    outputs={'_tmp'},
    code='_tmp = _a * _b'
)

# Connectors
outer_entry.add_in_connector('IN_A')
outer_entry.add_in_connector('IN_B')
outer_entry.add_out_connector('OUT_A')
outer_entry.add_out_connector('OUT_B')

inner_entry.add_in_connector('IN_A')
inner_entry.add_in_connector('IN_B')
inner_entry.add_out_connector('OUT_A')
inner_entry.add_out_connector('OUT_B')

inner_exit.add_in_connector('IN_C')
inner_exit.add_out_connector('OUT_C')

outer_exit.add_in_connector('IN_C')
outer_exit.add_out_connector('OUT_C')

# inputs -> outer_entry
state.add_edge(A_node, None, outer_entry, 'IN_A', dace.Memlet('A[0:M, 0:K]'))
state.add_edge(B_node, None, outer_entry, 'IN_B', dace.Memlet('B[0:K, 0:N]'))

# outer_entry -> inner_entry
state.add_edge(outer_entry, 'OUT_A', inner_entry, 'IN_A', dace.Memlet('A[i, 0:K]'))
state.add_edge(outer_entry, 'OUT_B', inner_entry, 'IN_B', dace.Memlet('B[0:K, j]'))

# inner_entry -> mac_tasklet
state.add_edge(inner_entry, 'OUT_A', mac_tasklet, '_a', dace.Memlet('A[i, k]'))
state.add_edge(inner_entry, 'OUT_B', mac_tasklet, '_b', dace.Memlet('B[k, j]'))

# mac_tasklet -> inner_exit
state.add_edge(mac_tasklet, '_tmp', inner_exit, 'IN_C', dace.Memlet(data='C', subset='i, j', wcr='lambda x, y: x + y'))

# inner_exit -> outer_exit
state.add_edge(inner_exit, 'OUT_C', outer_exit, 'IN_C', dace.Memlet('C[i, j]'))

# outer_exit -> C_node
state.add_edge(outer_exit, 'OUT_C', C_node, None, dace.Memlet('C[0:M, 0:N]'))

sdfg.validate()
sdfg.save('matmul_manual.sdfg')

compiled = sdfg.compile()
A = np.random.rand(64, 32)
B = np.random.rand(32, 64)
C = np.zeros((64, 64))
compiled(A=A, B=B, C=C, M=64, N=64, K=32)
expected = A @ B
print("Max error:", np.max(np.abs(C - expected)))
