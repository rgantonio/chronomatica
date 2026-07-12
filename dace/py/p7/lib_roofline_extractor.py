
import dace
import sympy
import re
import numpy as np
import glob
from dace.codegen.instrumentation.report import InstrumentationReport


# ============================================================
# Step 1: SDFG Inspector
# Run this first on any new SDFG to understand its structure
# before running the extractor.
# ============================================================

def inventory_sdfg(sdfg, indent=0):
    """
    Print all maps and tasklets with codes and WCR status.
    Always run before extract_roofline_data on a new workload.
    """
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                print(' ' * indent +
                      f"MAP: {node.label}  range={node.map.range}")
            if isinstance(node, dace.nodes.Tasklet):
                entry = state.entry_node(node)
                wcr = [e.data.wcr for e in state.out_edges(node)
                       if e.data.wcr]
                print(' ' * indent +
                      f"  TASKLET: {node.label}")
                print(' ' * indent +
                      f"    code: '{node.code.as_string.strip()}'")
                print(' ' * indent +
                      f"    map:  "
                      f"{entry.label if entry else 'none'}")
                if wcr:
                    print(' ' * indent + f"    wcr:  {wcr}")
            if isinstance(node, dace.nodes.NestedSDFG):
                print(' ' * indent +
                      f"[NestedSDFG: {node.label}]")
                inventory_sdfg(node.sdfg, indent + 4)


# ============================================================
# Step 2: FLOP counter
# ============================================================

def is_init_tasklet(code_str):
    """
    Detect pure assignment/init tasklets with no compute.
    A tasklet is init if it has no arithmetic operators AND
    no function calls (e.g. exp(), sqrt()).
    Note: always check WCR first — a pure assignment with a
    WCR output is still a real compute operation.
    """
    code = code_str.strip()
    has_arithmetic    = bool(re.search(r'[\+\-\*\/]', code))
    has_function_call = bool(re.search(r'\w+\s*\(', code))
    return not has_arithmetic and not has_function_call


def count_tasklet_ops(tasklet, state):
    """
    Count arithmetic ops in tasklet code AND WCR memlets.

    Fix 1: Check WCR first — reductions like identity + WCR
           look like init tasklets but perform real adds.
    Fix 2: Transcendental functions (exp, log, sqrt, etc.)
           count as 1 op each — they have no arithmetic operators
           so regex alone misses them.
    Fix 3: WCR lambda ops are on output edges, not in tasklet code.
    """
    code = tasklet.code.as_string.strip()

    # Check WCR on output edges first
    wcr_ops = 0
    for edge in state.out_edges(tasklet):
        if edge.data.wcr is not None:
            wcr_str = edge.data.wcr
            wcr_ops += len(re.findall(
                r'(?<![<>!])[\+\-](?!=)', wcr_str))
            wcr_ops += len(re.findall(r'\*(?!=)', wcr_str))

    # If pure assignment but has WCR — the WCR is the compute
    if is_init_tasklet(code):
        return wcr_ops   # 0 if truly init, >0 if WCR reduction

    # Count tasklet body ops
    multiplies      = len(re.findall(r'\*(?!=)', code))
    adds            = len(re.findall(r'(?<![<>!])[\+\-](?!=)', code))
    divides         = len(re.findall(r'/(?!=)', code))
    transcendentals = len(re.findall(
        r'\b(exp|log|sqrt|sin|cos|tan|tanh|sigmoid)\s*\(', code))

    tasklet_ops = multiplies + adds + divides + transcendentals
    return max(tasklet_ops + wcr_ops, 1)


def get_map_iterations(map_entry):
    """
    Compute total iteration count across all map dimensions.
    Fix: DaCe stores stop = N-1 for range 0:N, so the correct
    formula is (stop - start + 1) / step, not (stop - start) / step.
    """
    total = sympy.Integer(1)
    for r in map_entry.map.range:
        start, stop, step = r
        total = total * sympy.simplify((stop - start + 1) / step)
    return sympy.simplify(total)


def get_flops_for_map(node, state):
    """Count total FLOPs for a MapEntry node."""
    tasklet_ops = 0
    for inner_node in state.nodes():
        if isinstance(inner_node, dace.nodes.Tasklet):
            if state.entry_node(inner_node) == node:
                tasklet_ops += count_tasklet_ops(inner_node, state)
    if tasklet_ops == 0:
        return sympy.Integer(0)
    return sympy.simplify(get_map_iterations(node) * tasklet_ops)


# ============================================================
# Step 3: Bytes counter
# ============================================================

def get_subset_volume(subset):
    """Compute number of elements in a memlet subset."""
    volume = sympy.Integer(1)
    for dim in subset:
        start, stop, step = dim
        volume = volume * sympy.simplify((stop - start) / step + 1)
    return sympy.simplify(volume)


def get_bytes_for_map(node, state, current_sdfg):
    """
    Count bytes crossing map boundary edges (reads + writes).
    Fix: always pass current_sdfg (the SDFG that owns this state's
    arrays). When recursing into nested SDFGs, this must be the
    nested SDFG — not the top-level parent — or array lookups fail.
    """
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


# ============================================================
# Step 4: Collector — pure recursion, no printing
# ============================================================

def collect_roofline_data(sdfg, prefix='', counters=None):
    """
    Recursively collect roofline data from all maps.
    Returns flat dict: {label: {flops, bytes, intensity}}

    Fix 1: Separate collection from printing — printing inside
           recursion produces multiple partial tables.
    Fix 2: Deduplicate labels — two GEMMs with the same name
           overwrite each other without the counter.
    Fix 3: Symbolic comparison — never use > with SymPy
           expressions; use != sympy.Integer(0) instead.
    Fix 4: Pass prefix through recursion to build full path labels.
    """
    if counters is None:
        counters = {}

    results = {}

    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                flops       = get_flops_for_map(node, state)
                bytes_moved = get_bytes_for_map(node, state, sdfg)

                # Skip nodes with both zero FLOPs and zero bytes
                if (flops == sympy.Integer(0)
                        and bytes_moved == sympy.Integer(0)):
                    continue

                # Fix: never use > with SymPy — use structural check
                intensity = (
                    sympy.simplify(flops / bytes_moved)
                    if bytes_moved != sympy.Integer(0)
                    else sympy.oo
                )

                # Deduplicate identical labels (e.g. two gemm_maps)
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

            # Recurse into nested SDFGs with updated prefix
            if isinstance(node, dace.nodes.NestedSDFG):
                nested = collect_roofline_data(
                    node.sdfg,
                    prefix=f"{prefix}{node.label}/",
                    counters=counters
                )
                results.update(nested)

    return results


# ============================================================
# Step 5: Printer — called once at top level only
# ============================================================

def print_roofline_table(results, symbol_values):
    """
    Print formatted roofline table with concrete values.
    Call this once at the top level after collect_roofline_data.
    """
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


# ============================================================
# Step 6: Report reader (for timing data)
# ============================================================

def read_report(report_path):
    """
    Parse DaCe instrumentation report.
    Wall time per map = max across all threads per run
    (threads run simultaneously, not sequentially).
    Returns: {map_label: {mean_ms, min_ms, median_ms, node_id}}
    """
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


# ============================================================
# Step 7: Top-level entry points
# ============================================================

def extract_roofline_data(sdfg, symbol_values=None):
    """
    Collect roofline data and optionally print table.
    Main entry point for intensity-only analysis.
    """
    results = collect_roofline_data(sdfg)
    if symbol_values:
        print_roofline_table(results, symbol_values)
    return results


def full_roofline_analysis(sdfg_path, symbol_values,
                           report_path=None):
    """
    Full roofline analysis combining intensity + timing.
    Produces GFLOPs/s per component when report_path is provided.
    """
    sdfg = dace.SDFG.from_file(sdfg_path)
    results = collect_roofline_data(sdfg)
    timing_data = read_report(report_path) if report_path else {}

    has_timing = bool(report_path)
    width = 90 if has_timing else 75

    if has_timing:
        print(f"\n{'='*width}")
        print(f"{'Component':<45} {'GFLOPs':>10} {'MB':>8} "
              f"{'I (F/B)':>8} {'ms':>8} {'GF/s':>8}")
    else:
        print(f"\n{'='*width}")
        print(f"{'Component':<45} {'GFLOPs':>10} {'MB':>8} "
              f"{'I (F/B)':>10}")
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

