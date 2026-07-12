import dace
import sympy
import re
import numpy as np
import glob
from dace.codegen.instrumentation.report import InstrumentationReport

# ── FLOP counter ────────────────────────────────────────────────────

def is_init_tasklet(code_str):
    code = code_str.strip()
    has_arithmetic    = bool(re.search(r'[\+\-\*\/]', code))
    has_function_call = bool(re.search(r'\w+\s*\(', code))
    return not has_arithmetic and not has_function_call

def count_tasklet_ops(tasklet, state):
    code = tasklet.code.as_string.strip()
    wcr_ops = 0
    for edge in state.out_edges(tasklet):
        if edge.data.wcr is not None:
            wcr_str = edge.data.wcr
            wcr_ops += len(re.findall(
                r'(?<![<>!])[\+\-](?!=)', wcr_str))
            wcr_ops += len(re.findall(r'\*(?!=)', wcr_str))
    if is_init_tasklet(code):
        return wcr_ops
    multiplies      = len(re.findall(r'\*(?!=)', code))
    adds            = len(re.findall(r'(?<![<>!])[\+\-](?!=)', code))
    divides         = len(re.findall(r'/(?!=)', code))
    transcendentals = len(re.findall(
        r'\b(exp|log|sqrt|sin|cos|tan|tanh|sigmoid)\s*\(', code))
    tasklet_ops = multiplies + adds + divides + transcendentals
    return max(tasklet_ops + wcr_ops, 1)

def get_map_iterations(map_entry):
    total = sympy.Integer(1)
    for r in map_entry.map.range:
        start, stop, step = r
        total = total * sympy.simplify((stop - start + 1) / step)
    return sympy.simplify(total)

def get_flops_for_map(node, state):
    tasklet_ops = 0
    for inner_node in state.nodes():
        if isinstance(inner_node, dace.nodes.Tasklet):
            if state.entry_node(inner_node) == node:
                tasklet_ops += count_tasklet_ops(inner_node, state)
    if tasklet_ops == 0:
        return sympy.Integer(0)
    return sympy.simplify(get_map_iterations(node) * tasklet_ops)

# ── Bytes counter ────────────────────────────────────────────────────

def get_subset_volume(subset):
    volume = sympy.Integer(1)
    for dim in subset:
        start, stop, step = dim
        volume = volume * sympy.simplify((stop - start) / step + 1)
    return sympy.simplify(volume)

def get_bytes_for_map(node, state, current_sdfg):
    total_bytes = sympy.Integer(0)
    map_exit = state.exit_node(node)
    for edge in state.in_edges(node):
        m = edge.data
        if m.data is not None and m.subset is not None:
            if m.data in current_sdfg.arrays:
                arr = current_sdfg.arrays[m.data]
                total_bytes += (get_subset_volume(m.subset)
                                * arr.dtype.bytes)
    for edge in state.out_edges(map_exit):
        m = edge.data
        if m.data is not None and m.subset is not None:
            if m.data in current_sdfg.arrays:
                arr = current_sdfg.arrays[m.data]
                total_bytes += (get_subset_volume(m.subset)
                                * arr.dtype.bytes)
    return sympy.simplify(total_bytes)

# ── Collector ────────────────────────────────────────────────────────

def collect_roofline_data(sdfg, prefix='', counters=None):
    """
    Recursively collect roofline data. No printing — pure collection.
    Returns flat dict: {label: {flops, bytes, intensity}}
    counters: shared dict for deduplicating identical labels.
    """
    if counters is None:
        counters = {}

    results = {}

    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                flops       = get_flops_for_map(node, state)
                bytes_moved = get_bytes_for_map(node, state, sdfg)

                # Skip zero-FLOP zero-byte nodes (pure init maps)
                if (flops == sympy.Integer(0)
                        and bytes_moved == sympy.Integer(0)):
                    continue

                # Symbolic comparison fix — never use > with SymPy
                intensity = (sympy.simplify(flops / bytes_moved)
                             if bytes_moved != sympy.Integer(0)
                             else sympy.oo)

                # Deduplicate labels
                base_label = f"{prefix}{state.label}/{node.label}"
                count = counters.get(base_label, 0)
                counters[base_label] = count + 1
                label = (f"{base_label}_{count}"
                         if count > 0 else base_label)

                results[label] = {
                    'flops':     flops,
                    'bytes':     bytes_moved,
                    'intensity': intensity
                }

            if isinstance(node, dace.nodes.NestedSDFG):
                nested = collect_roofline_data(
                    node.sdfg,
                    prefix=f"{prefix}{node.label}/",
                    counters=counters
                )
                results.update(nested)

    return results

# ── Printer ──────────────────────────────────────────────────────────

def print_roofline_table(results, symbol_values):
    """Print formatted roofline table. Call once at top level only."""
    print(f"\n{'='*75}")
    print(f"{'Component':<45} {'GFLOPs':>10} {'MB':>8} {'I (F/B)':>10}")
    print(f"{'='*75}")

    total_flops = 0
    total_bytes = 0

    for label, data in results.items():
        try:
            f = int(data['flops'].subs(symbol_values))
            b = int(data['bytes'].subs(symbol_values))
            i = f / b if b > 0 else float('inf')
            total_flops += f
            total_bytes += b
            print(f"  {label:<43} "
                  f"{f/1e9:>10.6f} "
                  f"{b/1e6:>8.2f} "
                  f"{i:>10.4f}")
        except Exception as e:
            print(f"  {label}: (symbolic) {e}")

    print(f"{'='*75}")
    if total_bytes > 0:
        print(f"  {'TOTAL':<43} "
              f"{total_flops/1e9:>10.6f} "
              f"{total_bytes/1e6:>8.2f} "
              f"{total_flops/total_bytes:>10.4f}")
    else:
        print(f"  {'TOTAL':<43} "
              f"{total_flops/1e9:>10.6f} "
              f"{total_bytes/1e6:>8.2f} "
              f"{'N/A':>10}")
    print(f"{'='*75}")

# ── Report reader ─────────────────────────────────────────────────────

def read_report(report_path):
    """Parse instrumentation report into {map_label: timing_dict}."""
    report = InstrumentationReport(report_path)
    timings = {}
    for key, entry in report.durations.items():
        for label, thread_data in entry.items():
            clean_label = label.replace('Map ', '').strip()
            all_thread_lists = list(thread_data.values())
            n_runs = len(all_thread_lists[0])
            wall_times = []
            for run_idx in range(n_runs):
                run_max = max(
                    tl[run_idx]
                    for tl in all_thread_lists
                    if run_idx < len(tl)
                )
                wall_times.append(run_max)
            timings[clean_label] = {
                'mean_ms':   np.mean(wall_times),
                'min_ms':    np.min(wall_times),
                'median_ms': np.median(wall_times),
                'node_id':   key[2]
            }
    return timings

# ── Full analysis ─────────────────────────────────────────────────────

def extract_roofline_data(sdfg, symbol_values=None):
    """Collect and optionally print roofline data."""
    results = collect_roofline_data(sdfg)
    if symbol_values:
        print_roofline_table(results, symbol_values)
    return results

def full_roofline_analysis(sdfg_path, symbol_values,
                           report_path=None):
    """
    Complete roofline analysis combining intensity + timing.
    """
    sdfg = dace.SDFG.from_file(sdfg_path)
    results = collect_roofline_data(sdfg)
    timing_data = read_report(report_path) if report_path else {}

    has_timing = bool(report_path)
    width = 90 if has_timing else 75
    header = (f"{'Component':<45} {'GFLOPs':>10} {'MB':>8} "
              f"{'I (F/B)':>8} {'ms':>8} {'GF/s':>8}"
              if has_timing else
              f"{'Component':<45} {'GFLOPs':>10} {'MB':>8} {'I (F/B)':>10}")

    print(f"\n{'='*width}")
    print(header)
    print(f"{'='*width}")

    total_flops = 0
    total_bytes = 0

    for label, data in results.items():
        try:
            f = int(data['flops'].subs(symbol_values))
            b = int(data['bytes'].subs(symbol_values))
            i = f / b if b > 0 else float('inf')
            total_flops += f
            total_bytes += b

            if has_timing:
                short_label = re.sub(r'_\d+$', '',
                                     label.split('/')[-1])
                timing = None
                for tl, td in timing_data.items():
                    if short_label in tl:
                        if (timing is None or
                                td['node_id'] < timing['node_id']):
                            timing = td
                if timing:
                    t_ms = timing['mean_ms']
                    gfs  = f / 1e9 / (t_ms / 1000.0)
                    print(f"  {label:<43} {f/1e9:>10.6f} "
                          f"{b/1e6:>8.2f} {i:>8.4f} "
                          f"{t_ms:>8.3f} {gfs:>8.3f}")
                else:
                    print(f"  {label:<43} {f/1e9:>10.6f} "
                          f"{b/1e6:>8.2f} {i:>8.4f} "
                          f"{'N/A':>8} {'N/A':>8}")
            else:
                print(f"  {label:<43} {f/1e9:>10.6f} "
                      f"{b/1e6:>8.2f} {i:>10.4f}")

        except Exception as e:
            print(f"  {label}: (symbolic) {e}")

    print(f"{'='*width}")
    if total_bytes > 0:
        print(f"  {'TOTAL':<43} "
              f"{total_flops/1e9:>10.6f} "
              f"{total_bytes/1e6:>8.2f} "
              f"{total_flops/total_bytes:>10.4f}")
    else:
        print(f"  {'TOTAL':<43} "
              f"{total_flops/1e9:>10.6f} "
              f"{total_bytes/1e6:>8.2f} "
              f"{'N/A':>10}")
    print(f"{'='*width}")
    return results


# ── Test on matmul ────────────────────────────────────────────────────

if __name__ == '__main__':
    import dace
    import numpy as np

    M, N, K = dace.symbol('M'), dace.symbol('N'), dace.symbol('K')

    @dace.program
    def matmul(A: dace.float64[M, K], B: dace.float64[K, N]):
        return A @ B

    sdfg = matmul.to_sdfg()
    sdfg.simplify()
    sdfg.expand_library_nodes()
    sdfg.save('matmul_auto.sdfg')

    symbol_values = {'M': 512, 'N': 512, 'K': 256}

    print("=== Intensity only ===")
    extract_roofline_data(sdfg, symbol_values=symbol_values)
