import dace

sdfg = dace.SDFG.from_file('axpy_stream.sdfg')

# Set FPGA schedule on all maps
for state in sdfg.states():
    for node in state.nodes():
        if isinstance(node, dace.nodes.MapEntry):
            node.map.schedule = dace.ScheduleType.FPGA_Device

# Just compile directly
sdfg.compile()