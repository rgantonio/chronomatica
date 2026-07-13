import dace
import numpy as np

N = dace.symbol('N')
sdfg = dace.SDFG('axpy_stream')

# Declare arrays
sdfg.add_array('a',      shape=[1],  dtype=dace.float64,
               transient=False)
sdfg.add_array('x',      shape=[N],  dtype=dace.float64,
               transient=False)
sdfg.add_array('y',      shape=[N],  dtype=dace.float64,
               transient=False)
sdfg.add_array('result', shape=[N],  dtype=dace.float64,
               transient=False)

# Declare a stream — FIFO buffer of N float64 elements
# buffer_size controls how many elements can be in flight
sdfg.add_stream('tmp_stream',
                dtype=dace.float64,
                buffer_size=1,        # 1 = fully pipelined
                transient=True)

state = sdfg.add_state('compute')

# Access nodes
a_node      = state.add_read('a')
x_node      = state.add_read('x')
y_node      = state.add_read('y')
out_node    = state.add_write('result')
stream_node = state.add_access('tmp_stream')

# Map 1: multiply — producer, pushes into stream
# Scheduled sequentially: the two maps communicate through a FIFO
# stream, so push/pop order must match iteration order. OpenMP's
# default parallel scheduling would push/pop out of order and
# scramble the x[i]/y[i] pairing.
me1, mx1 = state.add_map('multiply_map', {'i': '0:N'},
                         schedule=dace.ScheduleType.Sequential)
mult_tasklet = state.add_tasklet(
    'multiply', {'_a', '_x'}, {'_out'}, '_out = _a * _x')

me1.add_in_connector('IN_a')
me1.add_in_connector('IN_x')
me1.add_out_connector('OUT_a')
me1.add_out_connector('OUT_x')
mx1.add_in_connector('IN_stream')
mx1.add_out_connector('OUT_stream')

state.add_edge(a_node,      None,    me1,         'IN_a',
               dace.Memlet('a[0]'))
state.add_edge(x_node,      None,    me1,         'IN_x',
               dace.Memlet('x[0:N]'))
state.add_edge(me1,         'OUT_a', mult_tasklet, '_a',
               dace.Memlet('a[0]'))
state.add_edge(me1,         'OUT_x', mult_tasklet, '_x',
               dace.Memlet('x[i]'))

# Push to stream — note the memlet uses stream name
state.add_edge(mult_tasklet, '_out', mx1,         'IN_stream',
               dace.Memlet('tmp_stream[0]'))
state.add_edge(mx1,         'OUT_stream', stream_node, None,
               dace.Memlet('tmp_stream[0:N]'))

# Map 2: add — consumer, pops from stream
me2, mx2 = state.add_map('add_map', {'i': '0:N'},
                         schedule=dace.ScheduleType.Sequential)
add_tasklet = state.add_tasklet(
    'add', {'_tmp', '_y'}, {'_out'}, '_out = _tmp + _y')

me2.add_in_connector('IN_stream')
me2.add_in_connector('IN_y')
me2.add_out_connector('OUT_stream')
me2.add_out_connector('OUT_y')
mx2.add_in_connector('IN_result')
mx2.add_out_connector('OUT_result')

state.add_edge(stream_node, None,    me2,         'IN_stream',
               dace.Memlet('tmp_stream[0:N]'))
state.add_edge(y_node,      None,    me2,         'IN_y',
               dace.Memlet('y[0:N]'))
state.add_edge(me2,         'OUT_stream', add_tasklet, '_tmp',
               dace.Memlet('tmp_stream[0]'))
state.add_edge(me2,         'OUT_y', add_tasklet, '_y',
               dace.Memlet('y[i]'))
state.add_edge(add_tasklet, '_out',  mx2,         'IN_result',
               dace.Memlet('result[i]'))
state.add_edge(mx2,         'OUT_result', out_node, None,
               dace.Memlet('result[0:N]'))

sdfg.validate()
sdfg.save('axpy_stream.sdfg')

compiled = sdfg.compile()

a = np.array([2.0])
x = np.random.rand(1024)
y = np.random.rand(1024)
result = np.zeros(1024)
compiled(a=a, x=x, y=y, result=result, N=1024)

expected = 2.0 * x + y
print("Max error:", np.max(np.abs(result - expected)))