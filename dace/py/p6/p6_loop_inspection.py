import dace

sdfg = dace.SDFG.from_file('loop_sum.sdfg')

print("=== Interstate Edges ===")
for edge in sdfg.edges():
    print(f"{edge.src.label} → {edge.dst.label}")
    print(f"  condition:   {edge.data.condition.as_string}")
    print(f"  assignments: {edge.data.assignments}")
    print()