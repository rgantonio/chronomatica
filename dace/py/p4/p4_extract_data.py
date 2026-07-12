import dace
import sympy

# default
M, N, K = dace.symbol('M'), dace.symbol('N'), dace.symbol('K')

@dace.program
def matmul(A: dace.float64[M, K], B: dace.float64[K, N]):
    return A @ B

sdfg = matmul.to_sdfg()
sdfg.simplify()
sdfg.expand_library_nodes()
sdfg.save('matmul_auto.sdfg')

sdfg = dace.SDFG.from_file('matmul_auto.sdfg')

def inspect_maps(sdfg, indent=0):
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                print(' ' * indent + f"Map: {node.label}")
                print(' ' * indent + f"  ranges: {node.map.range}")
                for i, r in enumerate(node.map.range):
                    start, stop, step = r
                    # Correct iteration count formula
                    count = (stop - start + 1) / step
                    print(' ' * indent +
                          f"  dim {i}: start={start} stop={stop} "
                          f"step={step}  iterations={count}")
            if isinstance(node, dace.nodes.Tasklet):
                print(' ' * indent + f"Tasklet: {node.label}")
                print(' ' * indent +
                      f"  code: {node.code.as_string}")
            if isinstance(node, dace.nodes.NestedSDFG):
                print(' ' * indent + f"[NestedSDFG: {node.label}]")
                inspect_maps(node.sdfg, indent + 4)

inspect_maps(sdfg)
