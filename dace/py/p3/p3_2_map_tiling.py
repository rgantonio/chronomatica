import dace
from dace.transformation.dataflow import MapTiling

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

# Step 1: Find all maps recursively
def find_all_maps(sdfg, indent=0):
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                print(' ' * indent +
                      f"Map: {node.label}  "
                      f"range: {node.map.range}  "
                      f"state: {state.label}")
            if isinstance(node, dace.nodes.NestedSDFG):
                print(' ' * indent + f"[NestedSDFG: {node.label}]")
                find_all_maps(node.sdfg, indent + 4)

find_all_maps(sdfg)

# Step 2: Tile only the gemm_map, not the init map
for state in sdfg.states():
    for node in state.nodes():
        if isinstance(node, dace.nodes.NestedSDFG):
            nested = node.sdfg
            for nested_state in nested.states():
                for n in nested_state.nodes():
                    if (isinstance(n, dace.nodes.MapEntry)
                            and n.label == 'gemm_map'):
                        print(f"Tiling: {n.label}  range: {n.map.range}")
                        nested.apply_transformations(
                            MapTiling,
                            options={'tile_sizes': [32, 32, 32]},
                            states=[nested_state]
                        )

sdfg.validate()
sdfg.save('matmul_tiled.sdfg')