import dace
import numpy as np

S = dace.symbol('S')   # sequence length
D = dace.symbol('D')   # head dimension

@dace.program
def self_attention(Q: dace.float64[S, D],
                   K: dace.float64[S, D],
                   V: dace.float64[S, D]):
    # Step 1: QK^T — shape [S, S]
    scores_qk = Q @ np.transpose(K)

    # Step 2: Scale — unique name avoids double AccessNode
    scores_scaled = scores_qk / np.sqrt(D)

    # Step 3: Softmax
    scores_exp  = np.exp(scores_scaled)
    scores_sum  = np.sum(scores_exp, axis=1)
    scores_norm = scores_exp / scores_sum[:, None]

    # Step 4: Weighted sum
    return scores_norm @ V

sdfg = self_attention.to_sdfg()
sdfg.simplify()
sdfg.save('self_attention.sdfg')