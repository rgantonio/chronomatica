import dace
from dace.transformation.dataflow import MapTiling

sdfg = dace.SDFG.from_file('self_attention_expanded.sdfg')

# Find and tile both gemm_maps
tiled_count = 0
for state in sdfg.states():
    for node in state.nodes():
        if isinstance(node, dace.nodes.NestedSDFG):
            if '_MatMult_gemm' in node.label:
                for nstate in node.sdfg.states():
                    for n in nstate.nodes():
                        if (isinstance(n, dace.nodes.MapEntry)
                                and n.label == 'gemm_map'):
                            print(f"Tiling GEMM: range={n.map.range}")
                            node.sdfg.apply_transformations(
                                MapTiling,
                                options={'tile_sizes': [32, 32, 32]},
                                states=[nstate]
                            )
                            tiled_count += 1

print(f"Tiled {tiled_count} GEMM maps")
sdfg.validate()
sdfg.save('self_attention_tiled.sdfg')