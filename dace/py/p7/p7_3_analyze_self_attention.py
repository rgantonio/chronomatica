import dace
import sympy
import re
import numpy as np
import glob
from dace.codegen.instrumentation.report import InstrumentationReport
from lib_roofline_extractor import extract_roofline_data, full_roofline_analysis

# ============================================================
# Usage examples
# ============================================================

if __name__ == '__main__':

    # ── Self-attention intensity analysis ───────────────────
    S = dace.symbol('S')
    D = dace.symbol('D')

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
    sdfg.expand_library_nodes()
    sdfg.save('self_attention_expanded.sdfg')

    print("=== Self-Attention Roofline Analysis ===")
    extract_roofline_data(
        sdfg,
        symbol_values={'S': 128, 'D': 64}
    )

    # ── With timing (uncomment if you have a report) ────────
    # report_files = sorted(glob.glob(
    #     '.dacecache/**/report-*.json', recursive=True))
    # if report_files:
    #     full_roofline_analysis(
    #         sdfg_path='self_attention_expanded.sdfg',
    #         symbol_values={'S': 128, 'D': 64},
    #         report_path=report_files[-1]
    #     )

    # ── Matmul sanity check ──────────────────────────────────
    M, N, K = dace.symbol('M'), dace.symbol('N'), dace.symbol('K')

    @dace.program
    def matmul(A: dace.float64[M, K], B: dace.float64[K, N]):
        return A @ B

    sdfg_mm = matmul.to_sdfg()
    sdfg_mm.simplify()
    sdfg_mm.expand_library_nodes()

    print("\n=== Matmul Sanity Check (expect I=32.0) ===")
    extract_roofline_data(
        sdfg_mm,
        symbol_values={'M': 512, 'N': 512, 'K': 256}
    )