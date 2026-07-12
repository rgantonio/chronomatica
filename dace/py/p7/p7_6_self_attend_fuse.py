import dace
from dace.transformation.dataflow import MapFusion
from lib_roofline_extractor import extract_roofline_data

sdfg = dace.SDFG.from_file('self_attention_expanded.sdfg')

# Apply MapFusion restricted to the top-level state
for state in sdfg.states():
    n_applied = sdfg.apply_transformations(
        MapFusion,
        states=[state]
    )
    if n_applied > 0:
        print(f"Applied {n_applied} fusions in {state.label}")

sdfg.validate()
sdfg.save('self_attention_div_exp_fused.sdfg')

print("\n=== Checking the fused self-attention ===")
extract_roofline_data(
    sdfg,
    symbol_values={'M': 512, 'N': 512, 'K': 256}
)