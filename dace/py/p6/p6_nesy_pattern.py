import dace
import numpy as np

N = dace.symbol('N')
sdfg = dace.SDFG('nesy_dispatch')

sdfg.add_array('features',  shape=[N], dtype=dace.float64)
sdfg.add_array('output',    shape=[N], dtype=dace.float64)
sdfg.add_scalar('confidence', dtype=dace.float64, transient=False)
sdfg.add_scalar('threshold',  dtype=dace.float64, transient=False)

# ── States ──────────────────────────────────────────────────────────
neural_state    = sdfg.add_state('neural_inference')
symbolic_guard  = sdfg.add_state('symbolic_guard')
symbolic_state  = sdfg.add_state('symbolic_reasoning')
merge_state     = sdfg.add_state('merge')

# ── Interstate edges ─────────────────────────────────────────────────
sdfg.add_edge(neural_state,   symbolic_guard,
              dace.InterstateEdge())   # always go to guard after neural
sdfg.add_edge(symbolic_guard, symbolic_state,
              dace.InterstateEdge(condition='confidence < threshold'))
sdfg.add_edge(symbolic_guard, merge_state,
              dace.InterstateEdge(condition='confidence >= threshold'))
sdfg.add_edge(symbolic_state, merge_state,
              dace.InterstateEdge())

# ── neural_inference: scale features (placeholder for real inference)
feat_read  = neural_state.add_read('features')
out_write  = neural_state.add_write('output')
me, mx     = neural_state.add_map('neural_map', {'i': '0:N'})

me.add_in_connector('IN_f');    me.add_out_connector('OUT_f')
mx.add_in_connector('IN_out');  mx.add_out_connector('OUT_out')

neural_tasklet = neural_state.add_tasklet(
    'neural_op', {'_f'}, {'_out'}, '_out = _f * 0.9')

neural_state.add_edge(feat_read,      None,     me,             'IN_f',
                      dace.Memlet('features[0:N]'))
neural_state.add_edge(me,             'OUT_f',  neural_tasklet, '_f',
                      dace.Memlet('features[i]'))
neural_state.add_edge(neural_tasklet, '_out',   mx,             'IN_out',
                      dace.Memlet('output[i]'))
neural_state.add_edge(mx,             'OUT_out', out_write,     None,
                      dace.Memlet('output[0:N]'))

# ── symbolic_reasoning: refine output (placeholder for real symbolic)
feat_read_s = symbolic_state.add_read('features')
out_read_s  = symbolic_state.add_read('output')
out_write_s = symbolic_state.add_write('output')

sym_tasklet = symbolic_state.add_tasklet(
    'symbolic_op',
    {'_f', '_out_in'},
    {'_out'},
    '_out = _out_in + (_f * 0.1)'   # symbolic correction
)

symbolic_state.add_edge(feat_read_s, None,       sym_tasklet, '_f',
                        dace.Memlet('features[0]'))
symbolic_state.add_edge(out_read_s,  None,       sym_tasklet, '_out_in',
                        dace.Memlet('output[0]'))
symbolic_state.add_edge(sym_tasklet, '_out',     out_write_s, None,
                        dace.Memlet('output[0]'))

sdfg.validate()
sdfg.save('nesy_dispatch.sdfg')

compiled = sdfg.compile()
features   = np.random.rand(64)
output     = np.zeros(64)
confidence = np.float64(0.3)   # low confidence → symbolic triggered
threshold  = np.float64(0.5)

compiled(features=features, output=output,
         confidence=confidence, threshold=threshold, N=64)

print("Symbolic path taken (confidence < threshold):",
      confidence < threshold)
print("Output[0]:", output[0])