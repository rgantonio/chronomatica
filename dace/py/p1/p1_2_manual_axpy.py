import dace
import numpy as np

# 1. Create an empty SDFG
N = dace.symbol('N')
sdfg = dace.SDFG('axpy_manual')

# 2. Declare arrays
sdfg.add_array('x', shape=[N], dtype=dace.float64)
sdfg.add_array('y', shape=[N], dtype=dace.float64)
sdfg.add_array('result', shape=[N], dtype=dace.float64)
sdfg.add_scalar('a', dtype=dace.float64, transient=False)

# 3. Add a state
state = sdfg.add_state('compute')

# 4. Add AccessNodes
x_node   = state.add_read('x')
y_node   = state.add_read('y')
a_node   = state.add_read('a')
out_node = state.add_write('result')

# 5. Add the Map (parallel loop over [0:N])
map_entry, map_exit = state.add_map('parallel_i', {'i': '0:N'})

map_entry.add_in_connector('IN_a')
map_entry.add_in_connector('IN_x')
map_entry.add_in_connector('IN_y')
map_entry.add_out_connector('OUT_a')
map_entry.add_out_connector('OUT_x')
map_entry.add_out_connector('OUT_y')

map_exit.add_in_connector('IN_result')
map_exit.add_out_connector('OUT_result')

# 6. Add a Tasklet
tasklet = state.add_tasklet(
    name='axpy_compute',
    inputs={'_a', '_x', '_y'},
    outputs={'_out'},
    code='_out = _a * _x + _y'
)

# 7. Connect with Memlets
state.add_edge(a_node, None, map_entry, 'IN_a',
               dace.Memlet(data='a', subset='0'))
state.add_edge(x_node, None, map_entry, 'IN_x',
               dace.Memlet('x[0:N]'))
state.add_edge(y_node, None, map_entry, 'IN_y',
               dace.Memlet('y[0:N]'))

state.add_edge(map_entry, 'OUT_a', tasklet, '_a',
               dace.Memlet(data='a', subset='0'))
state.add_edge(map_entry, 'OUT_x', tasklet, '_x',
               dace.Memlet('x[i]'))
state.add_edge(map_entry, 'OUT_y', tasklet, '_y',
               dace.Memlet('y[i]'))

state.add_edge(tasklet, '_out', map_exit, 'IN_result',
               dace.Memlet('result[i]'))
state.add_edge(map_exit, 'OUT_result', out_node, None,
               dace.Memlet('result[0:N]'))

# 8. Validate and save
sdfg.validate()
sdfg.save('axpy_manual.sdfg')

# 9. Compile and test
compiled = sdfg.compile()
a = np.float64(2.0)
x = np.random.rand(1024)
y = np.random.rand(1024)
result = np.zeros(1024)
compiled(a=a, x=x, y=y, result=result, N=1024)

expected = a * x + y
print("Max error:", np.max(np.abs(result - expected)))