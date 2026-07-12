import dace
from dace.transformation.dataflow import MapFusion

sdfg = dace.SDFG.from_file('self_attention_expanded.sdfg')

# Check out_degree of intermediates between _Div_ and _numpy_exp_
for state in sdfg.states():
    for node in state.nodes():
        if isinstance(node, dace.nodes.AccessNode):
            if node.data in ('__tmp3', 'scores'):
                out_deg = len(list(state.out_edges(node)))
                in_deg  = len(list(state.in_edges(node)))
                print(f"AccessNode: {node.data}  "
                      f"in_degree={in_deg}  "
                      f"out_degree={out_deg}")