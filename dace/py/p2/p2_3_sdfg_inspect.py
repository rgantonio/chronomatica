import dace

sdfg = dace.SDFG.from_file('matmul_auto_expand.sdfg')

def find_all_maps(sdfg, indent=0):
    """Recursively find all maps including inside nested SDFGs."""
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                print(' ' * indent +
                      f"Map: {node.label}  "
                      f"range: {node.map.range}  "
                      f"in state: {state.label}")
            if isinstance(node, dace.nodes.NestedSDFG):
                print(' ' * indent + f"[NestedSDFG: {node.label}]")
                find_all_maps(node.sdfg, indent + 4)

find_all_maps(sdfg)
