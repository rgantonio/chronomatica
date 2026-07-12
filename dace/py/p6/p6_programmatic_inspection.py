import dace

sdfg = dace.SDFG.from_file('conditional.sdfg')

print("=== States ===")
for state in sdfg.states():
    print(f"State: {state.label}")
    print(f"  nodes: {[type(n).__name__ for n in state.nodes()]}")

print("\n=== Interstate Edges ===")
for edge in sdfg.edges():
    print(f"{edge.src.label} → {edge.dst.label}")
    print(f"  condition:   {edge.data.condition.as_string}")
    print(f"  assignments: {edge.data.assignments}")
    print()