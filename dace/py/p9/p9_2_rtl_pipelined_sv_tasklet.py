import dace
import numpy as np
import os

N = dace.symbol('N')
sdfg = dace.SDFG('rtl_mac')

sdfg.add_array('a', shape=[N], dtype=dace.float32, transient=False)
sdfg.add_array('b', shape=[N], dtype=dace.float32, transient=False)
sdfg.add_array('c', shape=[N], dtype=dace.float32, transient=False)

state = sdfg.add_state('compute')

a_node = state.add_read('a')
b_node = state.add_read('b')
c_node = state.add_write('c')

me, mx = state.add_map('mac_map', {'i': '0:N'},
                       schedule=dace.ScheduleType.Sequential)

# SystemVerilog tasklet — you write the RTL here
tasklet = state.add_tasklet(
    name='mac_pipelined',
    inputs={'a', 'b'},
    outputs={'c'},
    code='''
        // 2-stage registered pipeline using actual DaCe port names
        // Stage 1 register
        logic [31:0] stage1_data;
        logic        stage1_valid;

        // Pipeline enable — only advance when downstream is ready
        wire pipe_en = m_axis_c_tready;

        // Accept inputs when pipeline can advance
        assign s_axis_a_tready = pipe_en;
        assign s_axis_b_tready = pipe_en;

        always_ff @(posedge ap_aclk) begin
            if (ap_areset) begin
                stage1_data  <= 32'b0;
                stage1_valid <= 1'b0;
                m_axis_c_tdata  <= 32'b0;
                m_axis_c_tvalid <= 1'b0;
                m_axis_c_tlast  <= 1'b0;
            end else if (pipe_en) begin
                // Stage 1: multiply
                stage1_data  <= s_axis_a_tdata * s_axis_b_tdata;
                stage1_valid <= s_axis_a_tvalid & s_axis_b_tvalid;

                // Stage 2: register output
                m_axis_c_tdata  <= stage1_data;
                m_axis_c_tvalid <= stage1_valid;
                m_axis_c_tlast  <= s_axis_a_tlast;
            end
        end

        assign m_axis_c_tkeep = 4'hF;
        assign ap_done = m_axis_c_tvalid & m_axis_c_tready & m_axis_c_tlast;
    ''',
    language=dace.dtypes.Language.SystemVerilog
)

me.add_in_connector('IN_a');  me.add_out_connector('OUT_a')
me.add_in_connector('IN_b');  me.add_out_connector('OUT_b')
mx.add_in_connector('IN_c');  mx.add_out_connector('OUT_c')

state.add_edge(a_node, None, me, 'IN_a', dace.Memlet('a[0:N]'))
state.add_edge(b_node, None, me, 'IN_b', dace.Memlet('b[0:N]'))
state.add_edge(me, 'OUT_a', tasklet, 'a', dace.Memlet('a[i]'))
state.add_edge(me, 'OUT_b', tasklet, 'b', dace.Memlet('b[i]'))
state.add_edge(tasklet, 'c', mx, 'IN_c', dace.Memlet('c[i]'))
state.add_edge(mx, 'OUT_c', c_node, None, dace.Memlet('c[0:N]'))

sdfg.validate()
sdfg.save('rtl_mac.sdfg')

# RTL codegen needs to unroll the map at codegen time, so N must be a
# known constant, not a free symbol.
sdfg.specialize({'N': 4})

# Generate code — don't compile to binary, just generate
program_code = sdfg.generate_code()

# generate_code() only returns CodeObjects in memory; it doesn't write
# them to disk. Write them out the same way sdfg.compile() would, but
# skip the actual CMake/verilator build.
from dace.codegen.compiler import generate_program_folder
generate_program_folder(sdfg, program_code, os.path.join('.dacecache', sdfg.name))

# Find and print the generated SystemVerilog
import glob

sv_files = glob.glob('.dacecache/**/*.sv', recursive=True)
v_files  = glob.glob('.dacecache/**/*.v',  recursive=True)
all_rtl  = sv_files + v_files

print(f"Found {len(all_rtl)} RTL files:")
for f in all_rtl:
    print(f"  {f}")
    print()
    with open(f, 'r') as fh:
        print(fh.read())
    print('='*60)
