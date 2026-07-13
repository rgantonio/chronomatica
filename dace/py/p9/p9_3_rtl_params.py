import dace
import re

def derive_snax_params(sdfg_path, symbol_values):
    """
    Derive SNAX shell parameters from a DaCe SDFG.
    Combines roofline analysis with RTL interface derivation.
    """
    sdfg = dace.SDFG.from_file(sdfg_path)
    params = {}

    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet):
                # Check if this is an RTL tasklet
                if node.language == dace.dtypes.Language.SystemVerilog:
                    print(f"RTL Tasklet: {node.label}")

                    # Data width from connected array dtype
                    for edge in state.in_edges(node):
                        if edge.data.data in sdfg.arrays:
                            arr = sdfg.arrays[edge.data.data]
                            params['data_width'] = arr.dtype.bytes * 8
                            break

                    # Number of input/output ports
                    params['num_input_ports']  = len(node.in_connectors)
                    params['num_output_ports'] = len(node.out_connectors)

                    # Pipeline stages from always_ff count
                    code = node.code.as_string
                    ff_count = len(re.findall(
                        r'always_ff', code))
                    params['num_pipeline_stages'] = max(ff_count, 1)

                    # Initiation interval — 1 if fully pipelined
                    params['initiation_interval'] = 1

                    print(f"  data_width:          {params.get('data_width', 'unknown')}")
                    print(f"  num_input_ports:     {params['num_input_ports']}")
                    print(f"  num_output_ports:    {params['num_output_ports']}")
                    print(f"  num_pipeline_stages: {params['num_pipeline_stages']}")
                    print(f"  initiation_interval: {params['initiation_interval']}")

            # Map range gives problem size
            if isinstance(node, dace.nodes.MapEntry):
                import sympy
                from sympy import Integer
                ranges = {}
                for i, r in enumerate(node.map.range):
                    start, stop, step = r
                    count = sympy.simplify(
                        (stop - start + 1) / step)
                    try:
                        ranges[f'dim_{i}'] = int(
                            count.subs(symbol_values))
                    except:
                        ranges[f'dim_{i}'] = str(count)
                params['problem_size'] = ranges
                print(f"  problem_size:        {ranges}")

    return params

# Test on our RTL MAC SDFG
params = derive_snax_params(
    'rtl_mac.sdfg',
    symbol_values={'N': 4}
)

print("\n=== SNAX Shell Parameters ===")
for k, v in params.items():
    print(f"  {k}: {v}")
