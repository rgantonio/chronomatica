import dace

sdfg = dace.SDFG.from_file('axpy.sdfg')

print(f"SDFG name: {sdfg.name}")
print(f"Symbols: {sdfg.symbols}")
print(f"Arrays: {list(sdfg.arrays.keys())}")
print()

for state in sdfg.states():
    print(f"=== State: {state.label} ===")
    for node in state.nodes():
        print(f"  {type(node).__name__:20s}  {node.label}")
        for edge in state.out_edges(node):
            print(f"    {edge.src.label} → {edge.dst.label}  "
                  f"memlet: {edge.data}  "
                  f"subset: {edge.data.subset}  "
                  f"volume: {edge.data.volume}")
    print()