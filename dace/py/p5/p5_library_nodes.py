import dace
from dace import library, nodes, dtypes
from dace.transformation.transformation import ExpandTransformation

@dace.library.expansion
class ExpandMACPure(ExpandTransformation):
    """
    Pure CPU expansion — explicit maps and tasklets.
    Used for correctness validation and profiling.
    """
    environments = []   # no external dependencies

    @staticmethod
    def expansion(node, state, sdfg):
        """
        Called by expand_library_nodes().
        Receives the abstract node and returns a concrete nested SDFG.
        node.n gives the problem size declared on the Library Node.
        """
        N = node.n

        nsdfg = dace.SDFG('mac_pure')
        nsdfg.add_array('_x', shape=[N], dtype=dace.float64)
        nsdfg.add_array('_y', shape=[N], dtype=dace.float64)
        nsdfg.add_array('_result', shape=[1], dtype=dace.float64)

        nstate = nsdfg.add_state('mac_compute')

        x_node = nstate.add_read('_x')
        y_node = nstate.add_read('_y')
        r_node = nstate.add_write('_result')

        me, mx = nstate.add_map('mac_map', {'i': f'0:{N}'})

        tasklet = nstate.add_tasklet(
            'mac_op',
            {'_xi', '_yi'},
            {'_out'},
            '_out = _xi * _yi'
        )

        me.add_in_connector('IN_x')
        me.add_in_connector('IN_y')
        me.add_out_connector('OUT_x')
        me.add_out_connector('OUT_y')
        mx.add_in_connector('IN_result')
        mx.add_out_connector('OUT_result')

        nstate.add_edge(x_node, None, me, 'IN_x',
                        dace.Memlet(f'_x[0:{N}]'))
        nstate.add_edge(y_node, None, me, 'IN_y',
                        dace.Memlet(f'_y[0:{N}]'))
        nstate.add_edge(me, 'OUT_x', tasklet, '_xi',
                        dace.Memlet('_x[i]'))
        nstate.add_edge(me, 'OUT_y', tasklet, '_yi',
                        dace.Memlet('_y[i]'))
        nstate.add_edge(tasklet, '_out', mx, 'IN_result',
                        dace.Memlet(data='_result', subset='0',
                                    wcr='lambda a, b: a + b'))
        nstate.add_edge(mx, 'OUT_result', r_node, None,
                        dace.Memlet('_result[0]'))

        return nsdfg
    
@dace.library.expansion
class ExpandMACAccelerator(ExpandTransformation):
    """
    Accelerator expansion — emits MMIO calls to your CVA6 accelerator.
    Replace 0xDEADBEEF with your actual hardware register address.
    """
    environments = []

    @staticmethod
    def expansion(node, state, sdfg):
        N = node.n

        nsdfg = dace.SDFG('mac_accelerator')
        nsdfg.add_array('_x', shape=[N], dtype=dace.float64)
        nsdfg.add_array('_y', shape=[N], dtype=dace.float64)
        nsdfg.add_array('_result', shape=[1], dtype=dace.float64)

        nstate = nsdfg.add_state('mac_accel')

        x_node = nstate.add_read('_x')
        y_node = nstate.add_read('_y')
        r_node = nstate.add_write('_result')

        # Single tasklet emitting raw C++ MMIO code
        tasklet = nstate.add_tasklet(
            'accel_call',
            {'_x_ptr', '_y_ptr'},
            {'_result_ptr'},
            # This string is emitted VERBATIM into generated C++
            # Replace with your actual CVA6 MMIO register addresses
            f'''
            volatile uint64_t* accel =
                (volatile uint64_t*)0xDEADBEEF;
            accel[0] = (uint64_t)_x_ptr;      // input X address
            accel[1] = (uint64_t)_y_ptr;      // input Y address
            accel[2] = (uint64_t){N};          // vector length
            accel[3] = 1;                      // start signal
            while(accel[4] != 1);              // poll until done
            *_result_ptr = *(double*)accel[5]; // read result
            ''',
            language=dace.dtypes.Language.CPP
        )

        nstate.add_edge(x_node, None, tasklet, '_x_ptr',
                        dace.Memlet(f'_x[0:{N}]'))
        nstate.add_edge(y_node, None, tasklet, '_y_ptr',
                        dace.Memlet(f'_y[0:{N}]'))
        nstate.add_edge(tasklet, '_result_ptr', r_node, None,
                        dace.Memlet('_result[0]'))

        return nsdfg
    
@dace.library.node
class MACNode(dace.nodes.LibraryNode):
    """
    Multiply-Accumulate: result = sum(x * y)
    Supports multiple implementations for design space exploration.
    """

    # Registry of available implementations
    implementations = {
        'pure':        ExpandMACPure,
        'accelerator': ExpandMACAccelerator
    }
    default_implementation = 'pure'

    # Node parameters — travel with the node in the SDFG,
    # serialized when saved, restored when loaded
    n = dace.properties.SymbolicProperty(default=1)

    def __init__(self, n, *args, **kwargs):
        super().__init__('MAC', *args, **kwargs)
        self.n = n
        # Declare the node's interface connectors
        self.add_in_connector('_x')
        self.add_in_connector('_y')
        self.add_out_connector('_result')

def build_mac_sdfg(implementation='pure'):
    N = 1024

    sdfg = dace.SDFG('mac_test')
    sdfg.add_array('x',      shape=[N], dtype=dace.float64)
    sdfg.add_array('y',      shape=[N], dtype=dace.float64)
    sdfg.add_array('result', shape=[1], dtype=dace.float64)

    state = sdfg.add_state('compute')

    x_node = state.add_read('x')
    y_node = state.add_read('y')
    r_node = state.add_write('result')

    # Instantiate the Library Node
    mac_node = MACNode(n=N)
    mac_node.implementation = implementation  # swap here
    state.add_node(mac_node)

    # Connect using the node's declared connectors
    state.add_edge(x_node, None, mac_node, '_x',
                   dace.Memlet(f'x[0:{N}]'))
    state.add_edge(y_node, None, mac_node, '_y',
                   dace.Memlet(f'y[0:{N}]'))
    state.add_edge(mac_node, '_result', r_node, None,
                   dace.Memlet('result[0]'))

    # Save unexpanded — shows the hexagon in the viewer
    sdfg.save(f'mac_{implementation}_unexpanded.sdfg')

    # Expand before compiling — replaces hexagon with nested SDFG
    sdfg.expand_library_nodes()
    sdfg.validate()
    sdfg.save(f'mac_{implementation}_expanded.sdfg')

    return sdfg


# Test pure implementation
import numpy as np

sdfg = build_mac_sdfg('pure')
compiled = sdfg.compile()

x = np.random.rand(1024)
y = np.random.rand(1024)
result = np.zeros(1)

compiled(x=x, y=y, result=result)
expected = np.dot(x, y)
print(f"Result:   {result[0]:.6f}")
print(f"Expected: {expected:.6f}")
print(f"Error:    {abs(result[0] - expected):.2e}")


def test_mac_view_only(implementation='accelerator'):
    N = 1024
    sdfg = dace.SDFG('mac_test')
    sdfg.add_array('x',      shape=[N], dtype=dace.float64)
    sdfg.add_array('y',      shape=[N], dtype=dace.float64)
    sdfg.add_array('result', shape=[1], dtype=dace.float64)

    state = sdfg.add_state('compute')
    x_node = state.add_read('x')
    y_node = state.add_read('y')
    r_node = state.add_write('result')

    mac_node = MACNode(n=N)
    mac_node.implementation = implementation
    state.add_node(mac_node)

    state.add_edge(x_node, None, mac_node, '_x',
                   dace.Memlet(f'x[0:{N}]'))
    state.add_edge(y_node, None, mac_node, '_y',
                   dace.Memlet(f'y[0:{N}]'))
    state.add_edge(mac_node, '_result', r_node, None,
                   dace.Memlet('result[0]'))

    # Save unexpanded — hexagon view
    sdfg.save(f'mac_{implementation}_unexpanded.sdfg')

    # Expand — replaces hexagon with MMIO nested SDFG
    sdfg.expand_library_nodes()
    sdfg.save(f'mac_{implementation}_expanded.sdfg')

    # View only — no compile, no validate
    return sdfg

test_mac_view_only('accelerator')
