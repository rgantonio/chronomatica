import dace
import re

sdfg = dace.SDFG.from_file('self_attention.sdfg')
sdfg.expand_library_nodes()
sdfg.save('self_attention_expanded.sdfg')

def inventory_sdfg(sdfg, indent=0):
    """
    Print all maps and tasklets with their codes and WCR status.
    Run this before the extractor to understand what patterns exist.
    """
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                print(' ' * indent +
                      f"MAP: {node.label}  range={node.map.range}")
            if isinstance(node, dace.nodes.Tasklet):
                entry = state.entry_node(node)
                wcr = [e.data.wcr for e in state.out_edges(node)
                       if e.data.wcr]
                print(' ' * indent +
                      f"  TASKLET: {node.label}")
                print(' ' * indent +
                      f"    code: '{node.code.as_string.strip()}'")
                print(' ' * indent +
                      f"    map:  {entry.label if entry else 'none'}")
                if wcr:
                    print(' ' * indent + f"    wcr:  {wcr}")
            if isinstance(node, dace.nodes.NestedSDFG):
                print(' ' * indent +
                      f"[NestedSDFG: {node.label}]")
                inventory_sdfg(node.sdfg, indent + 4)

inventory_sdfg(sdfg)