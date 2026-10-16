#!/usr/bin/env python3
"""
SKQD Benchmark Report Generator
Reads skqd_summary_statistics.csv and skqd_benchmark_results.csv
and generates a publication-grade markdown summary report (SKQD_BENCHMARK_REPORT.md)
with evaluation-budget-matched comparative tables stratified by trap depth (E0).
"""

import os
import sys
import csv
from collections import defaultdict

def generate_report(summary_csv_path="skqd_summary_statistics.csv",
                    detailed_csv_path="skqd_benchmark_results.csv",
                    output_md_path="SKQD_BENCHMARK_REPORT.md"):
    if not os.path.exists(summary_csv_path):
        print(f"Error: Could not find summary CSV file at '{summary_csv_path}'.")
        print("Please run the benchmark executable (./fixed_g_custatevec) first.")
        return False

    rows = []
    with open(summary_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("N") and not r["N"].startswith("#"):
                rows.append(r)

    if not rows:
        print(f"Warning: '{summary_csv_path}' has no data rows.")
        return False

    is_delayed_window_format = "TrapRank" in rows[0] and "M1_SuccessRate" in rows[0]
    is_per_trap_format = "TrapRank" in rows[0] and "Quench_MinE" in rows[0]

    # Group by scan configuration (N, g)
    scans = defaultdict(list)
    for r in rows:
        key = (
            int(r.get("N", 16)),
            float(r.get("g", 0.5)),
            int(r.get("K", 20 if is_delayed_window_format else 5)),
            int(r.get("M", 100 if is_delayed_window_format else 400)),
            float(r.get("dt", 0.05 if is_delayed_window_format else 0.1))
        )
        scans[key].append(r)

    has_standard_rates = "SKQD_PerEvalRate" in rows[0]

    print("\n" + "="*160)
    print(" "*40 + "SKQD BUDGET-MATCHED BENCHMARK: QUANTUM VS. CLASSICAL REPORT")
    print("="*160)

    if is_delayed_window_format:
        for (n, g, k, m, dt), trap_rows in scans.items():
            print(f"\n>>> DELAYED WINDOW BENCHMARK (g={g:.2f}): System Size N={n} | K={k}, M={m}, dt={dt:.4f} | Budget = 2000 evals")
            print("-" * 145)
            print(f"  {'Seed Rank':<15} | {'E0 (Opt E*)':<13} | {'M1 Succ %':>10} | {'Window <P_<':>12} | {'Best d*':>8} | {'M3(d*) Succ%':>12} | {'M3(d*) Rate':>11} | {'Restart Rate':>12} | {'<dH>':>6}")
            print("-" * 145)

            for tr in trap_rows:
                rank = tr.get("TrapRank", "")
                e0 = tr.get("SeedEnergy", "")
                opt_e = tr.get("KnownGlobalOpt", "")
                flips = float(tr.get("MeanFlips", 0.0))
                pless = float(tr.get("StatevecWindowPless", 0.0))
                m1_succ = float(tr.get("M1_SuccessRate", 0.0))
                best_d = tr.get("M3_Best_d", "")
                m3_best_succ = float(tr.get("M3_Best_SuccessRate", 0.0))
                m3_r = f"{float(tr.get('M3_Best_PerEvalRate', 0.0)):.2e}"
                rst_r = f"{float(tr.get('Restart_PerEvalRate', 0.0)):.2e}"

                trap_str = f"Seed #{rank}"
                seed_str = f"E0={e0} (E*={opt_e})"
                print(f"  {trap_str:<15} | {seed_str:<13} | {m1_succ:>9.1f}% | {pless:>12.4f} | {best_d:>8} | {m3_best_succ:>11.1f}% | {m3_r:>11} | {rst_r:>12} | {flips:>6.1f}")
            print("  " + "-"*143)

            # Energy level roll-up summary
            level_groups = defaultdict(list)
            for tr in trap_rows:
                level_groups[int(tr.get("SeedEnergy", 0))].append(tr)

            print("  --- SUMMARY ROLL-UP BY BOTTLENECK ENERGY LEVEL (Mean across symmetry families) ---")
            print(f"  {'Energy E0':<15} | {'Orbits':<13} | {'M1 Succ %':>10} | {'Window <P_<':>12} | {'M3(d*) Succ%':>12} | {'M3(match) %':>12}")
            print("  " + "-" * 115)
            for e_lvl in sorted(level_groups.keys()):
                group = level_groups[e_lvl]
                cnt = len(group)
                avg_m1 = sum(float(x.get("M1_SuccessRate", 0.0)) for x in group) / cnt
                avg_pless = sum(float(x.get("StatevecWindowPless", 0.0)) for x in group) / cnt
                avg_m3_best = sum(float(x.get("M3_Best_SuccessRate", 0.0)) for x in group) / cnt
                avg_m3_match = sum(float(x.get("M3_Matched_SuccessRate", 0.0)) for x in group) / cnt
            print("  " + "-" * 115)

        # Multi-g summary table across all swept g values for each N
        n_systems = sorted(list(set(n for (n, g, k, m, dt) in scans.keys())))
        for n_val in n_systems:
            g_vals = sorted(list(set(g for (n, g, k, m, dt) in scans.keys() if n == n_val)))
            if len(g_vals) > 1:
                g_str = ", ".join(f"{gv:.2f}" for gv in g_vals)
                print(f"\n>>> MULTI-g TRANSVERSE FIELD SCAN SUMMARY (N={n_val}): g in {{{g_str}}}")
                print("-" * 145)
                g_succ_cols = " | ".join(f"{f'g={gv:.2f}':>10}" for gv in g_vals)
                print(f"  {'Seed Rank':<15} | {'E0 (Opt E*)':<13} | {g_succ_cols} | {'Best d*':>8} | {'M3(d*) Succ%':>12} | {'Restart Succ%':>13}")
                print("-" * 145)

                rank_to_g = defaultdict(dict)
                for (n, g, k, m, dt), trap_rows in scans.items():
                    if n == n_val:
                        for tr in trap_rows:
                            rank = int(tr.get("TrapRank", 0))
                            rank_to_g[rank][g] = tr

                for rank in sorted(rank_to_g.keys()):
                    g_map = rank_to_g[rank]
                    sample_tr = next(iter(g_map.values()))
                    e0 = sample_tr.get("SeedEnergy", "")
                    opt_e = sample_tr.get("KnownGlobalOpt", "")
                    best_d = sample_tr.get("M3_Best_d", "")
                    m3_succ = float(sample_tr.get("M3_Best_SuccessRate", 0.0))
                    rst_succ = float(sample_tr.get("Restart_SuccessRate", 0.0))

                    succ_vals = []
                    for gv in g_vals:
                        if gv in g_map:
                            succ_vals.append(f"{float(g_map[gv].get('M1_SuccessRate', 0.0)):>9.1f}%")
                        else:
                            succ_vals.append(f"{'N/A':>10}")
                    succ_str = " | ".join(succ_vals)

                    trap_str = f"Seed #{rank}"
                    seed_str = f"E0={e0} (E*={opt_e})"
                    print(f"  {trap_str:<15} | {seed_str:<13} | {succ_str} | {best_d:>8} | {m3_succ:>11.1f}% | {rst_succ:>12.1f}%")
                print("  " + "-" * 143)
    elif is_per_trap_format:
        for (n, g, k, m, dt), trap_rows in scans.items():
            print(f"\n>>> CONFIGURATION: System Size N={n} | g={g:.2f} | K={k}, M={m}, dt={dt:.4f} (Quantum Shots: {k*m})")
            print("-" * 155)
            if has_standard_rates:
                print(f"  {'Trap Rank':<10} | {'Seed E0 (Opt E*)':<16} | {'Algorithm':<36} | {'Min E':>5} | {'Global Opt %':>12} | {'Success Rate %':>14} | {'Per-Eval Rate r':>15} | {'Mean Evals':>10}")
            else:
                print(f"  {'Trap Rank':<10} | {'Seed E0 (Opt E*)':<16} | {'Algorithm':<36} | {'Min E':>5} | {'Global Opt %':>12} | {'Success Rate %':>14} | {'Mean Drop ΔE':>12} | {'Mean Evals':>10}")
            print("-" * 155)

            for tr in trap_rows:
                rank = tr.get("TrapRank", "")
                e0 = tr.get("SeedEnergy", "")
                opt_e = tr.get("KnownGlobalOpt", "")
                flips = float(tr.get("MeanFlips", 4.0))
                kick_d = max(1, int(round(flips)))
                seed_str = f"E0={e0} (E*={opt_e})"

                algos = [
                    ("1. SKQD Dynamics (Raw Subspace)", "SKQD"),
                    ("2. Dynamics + SD Quench", "Quench"),
                    (f"3. Random Kick ({kick_d} flips) + SD", "Kick"),
                    ("4. Multi-Start SD (Random Restarts)", "Restart"),
                    ("5. Blind Random Sampling", "Blind")
                ]

                for idx, (display_name, prefix) in enumerate(algos):
                    rank_col = f"Level #{rank}" if idx == 0 else ""
                    seed_col = seed_str if idx == 0 else ""
                    min_e = tr.get(f"{prefix}_MinE", "")
                    opt_pct = f"{float(tr.get(f'{prefix}_OptPct', 0.0)):.1f}%"
                    succ_pct = f"{float(tr.get(f'{prefix}_SuccPct', 0.0)):.1f}%"
                    drop = f"{float(tr.get(f'{prefix}_MeanDrop', 0.0)):+.2f}"
                    evals = f"{float(tr.get(f'{prefix}_MeanEvals', 0.0)):.0f}"
                    r_col = f"{float(tr.get(f'{prefix}_PerEvalRate', 0.0)):.2e}" if has_standard_rates else drop

                    print(f"  {rank_col:<10} | {seed_col:<16} | {display_name:<36} | {min_e:>5} | {opt_pct:>12} | {succ_pct:>14} | {r_col:>15} | {evals:>10}")
                print("  " + "-"*153)
    else:
        # Fallback for legacy format
        print(f"  {'N':>2} | {'g':>5} | {'K':>2} {'M':>4} {'dt':>6} | {'Min E':>6} | {'Opt E*':>6} | {'Opt %':>7} | {'Mean Drop ΔE':>12} | {'Success %':>10}")
        for r in rows:
            print(f"  {r.get('N'):>2} | {float(r.get('g', 2.0)):>5.2f} | {r.get('K'):>2} {r.get('M'):>4} {float(r.get('dt', 0.0)):>6.4f} | {r.get('BestEnergyFound'):>6} | {r.get('KnownGlobalOpt'):>6} | {float(r.get('GlobalOptRate_Pct', 0.0)):>6.1f}% | {float(r.get('MeanEnergyDrop', 0.0)):>12.2f} | {float(r.get('SuccessRate_Pct', 0.0)):>9.1f}%")

    print("="*160 + "\n")

    # Generate Markdown Report (Pure Data Tables Only, No Prose)
    with open(output_md_path, mode="w", encoding="utf-8") as f:
        if is_delayed_window_format:
            f.write("# Delayed Window Quantum Dynamics Benchmark Results\n\n")
            for (n, g, k, m, dt), trap_rows in scans.items():
                f.write(f"## System Size $N = {n}$, Field $g = {g:.2f}$ (Delayed Window: $t_k \\in [10.0, 38.5]$, $K={k}, M={m}, \\Delta t={dt:.4f}$, Budget = 2000 evals)\n\n")
                f.write("| Seed Rank | Seed $E_0$ ($E^*$) | M1 Succ % | Window $\\langle P_< \\rangle$ | Best $d^*$ | M3($d^*$) Succ % | M3($d^*$) Rate ($ev^{-1}$) | Restart Rate ($ev^{-1}$) | Mean $\\langle d_H \\rangle$ |\n")
                f.write("|:---------:|:------------------:|:---------:|:-----------------------------:|:----------:|:-----------------:|:--------------------------:|:-------------------------:|:--------------------------:|\n")
                for tr in trap_rows:
                    rank = tr.get("TrapRank", "")
                    e0 = tr.get("SeedEnergy", "")
                    opt_e = tr.get("KnownGlobalOpt", "")
                    flips = float(tr.get("MeanFlips", 0.0))
                    pless = float(tr.get("StatevecWindowPless", 0.0))
                    m1_succ = float(tr.get("M1_SuccessRate", 0.0))
                    best_d = tr.get("M3_Best_d", "")
                    m3_best_succ = float(tr.get("M3_Best_SuccessRate", 0.0))
                    m3_r = f"{float(tr.get('M3_Best_PerEvalRate', 0.0)):.2e}"
                    rst_r = f"{float(tr.get('Restart_PerEvalRate', 0.0)):.2e}"
                    f.write(f"| Seed #{rank} | **$E_0={e0}$** ($E^*={opt_e}$) | {m1_succ:.1f}% | {pless:.4f} | {best_d} | {m3_best_succ:.1f}% | {m3_r} | {rst_r} | {flips:.1f} |\n")
                f.write("\n")

                # Markdown roll-up table
                level_groups = defaultdict(list)
                for tr in trap_rows:
                    level_groups[int(tr.get("SeedEnergy", 0))].append(tr)

                f.write("### Bottleneck Energy Level Roll-Up Summary (Mean across symmetry families)\n\n")
                f.write("| Energy $E_0$ | Distinct Orbits | Mean M1 Succ % | Mean Window $\\langle P_< \\rangle$ | Mean M3($d^*$) Succ % | Mean M3(match) Succ % |\n")
                f.write("|:------------:|:---------------:|:--------------:|:----------------------------------:|:---------------------:|:----------------------:|\n")
                for e_lvl in sorted(level_groups.keys()):
                    group = level_groups[e_lvl]
                    cnt = len(group)
                    avg_m1 = sum(float(x.get("M1_SuccessRate", 0.0)) for x in group) / cnt
                    avg_pless = sum(float(x.get("StatevecWindowPless", 0.0)) for x in group) / cnt
                    avg_m3_best = sum(float(x.get("M3_Best_SuccessRate", 0.0)) for x in group) / cnt
                    avg_m3_match = sum(float(x.get("M3_Matched_SuccessRate", 0.0)) for x in group) / cnt
                    f.write(f"| **$E_0={e_lvl}$** | {cnt} | {avg_m1:.1f}% | {avg_pless:.4f} | {avg_m3_best:.1f}% | {avg_m3_match:.1f}% |\n")
                f.write("\n")

                f.write("### Method 3 Kick Distance Scan Breakdown ($d \\in \\{2, 4, 6, 8, 10, 12\\}$)\n\n")
                f.write("| Trap Level | Seed $E_0$ | $d=2$ Succ % | $d=4$ Succ % | $d=6$ Succ % | $d=8$ Succ % | $d=10$ Succ % | $d=12$ Succ % |\n")
                f.write("|:----------:|:----------:|:------------:|:------------:|:------------:|:------------:|:-------------:|:-------------:|\n")
                for tr in trap_rows:
                    rank = tr.get("TrapRank", "")
                    e0 = tr.get("SeedEnergy", "")
                    d2 = float(tr.get("M3_d2_SuccessRate", 0.0))
                    d4 = float(tr.get("M3_d4_SuccessRate", 0.0))
                    d6 = float(tr.get("M3_d6_SuccessRate", 0.0))
                    d8 = float(tr.get("M3_d8_SuccessRate", 0.0))
                    d10 = float(tr.get("M3_d10_SuccessRate", 0.0))
                    d12 = float(tr.get("M3_d12_SuccessRate", 0.0))
                    f.write(f"| Level #{rank} | $E_0={e0}$ | {d2:.1f}% | {d4:.1f}% | {d6:.1f}% | {d8:.1f}% | {d10:.1f}% | {d12:.1f}% |\n")
                f.write("\n")

            # Multi-g summary table in Markdown across all swept g values for each N
            n_systems = sorted(list(set(n for (n, g, k, m, dt) in scans.keys())))
            for n_val in n_systems:
                g_vals = sorted(list(set(g for (n, g, k, m, dt) in scans.keys() if n == n_val)))
                if len(g_vals) > 1:
                    f.write(f"## System Size $N = {n_val}$: Cross-$g$ Transverse Field Comparison\n\n")
                    f.write(f"Comparative analysis of Method 1 across transverse field coupling strengths $g \\in \\{{{', '.join(f'{gv:.2f}' for gv in g_vals)}\\}}$ with evaluation budget $B = 2000$.\n\n")

                    # Table 1: M1 Success Rate vs g
                    f.write("### Method 1 Escape Success Rate vs. Transverse Field $g$\n\n")
                    g_headers = " | ".join(f"$g={gv:.2f}$" for gv in g_vals)
                    g_aligns = " | ".join([":---------:"] * len(g_vals))
                    f.write(f"| Seed Rank | Seed $E_0$ ($E^*$) | {g_headers} | Best $d^*$ | M3($d^*$) Succ % | Restart Succ % |\n")
                    f.write(f"|:---------:|:------------------:| {g_aligns} |:----------:|:-----------------:|:--------------:|\n")

                    rank_to_g = defaultdict(dict)
                    for (n, g, k, m, dt), trap_rows in scans.items():
                        if n == n_val:
                            for tr in trap_rows:
                                rank = int(tr.get("TrapRank", 0))
                                rank_to_g[rank][g] = tr

                    for rank in sorted(rank_to_g.keys()):
                        g_map = rank_to_g[rank]
                        sample_tr = next(iter(g_map.values()))
                        e0 = sample_tr.get("SeedEnergy", "")
                        opt_e = sample_tr.get("KnownGlobalOpt", "")
                        best_d = sample_tr.get("M3_Best_d", "")
                        m3_succ = float(sample_tr.get("M3_Best_SuccessRate", 0.0))
                        rst_succ = float(sample_tr.get("Restart_SuccessRate", 0.0))

                        m1_cells = " | ".join(f"{float(g_map[gv].get('M1_SuccessRate', 0.0)):.1f}%" if gv in g_map else "N/A" for gv in g_vals)
                        f.write(f"| Seed #{rank} | **$E_0={e0}$** ($E^*={opt_e}$) | {m1_cells} | {best_d} | {m3_succ:.1f}% | {rst_succ:.1f}% |\n")
                    f.write("\n")

                    # Table 2: Window <P_<> vs g
                    f.write("### Method 1 Window Escape Probability $\\langle P_< \\rangle$ vs. Transverse Field $g$\n\n")
                    f.write(f"| Seed Rank | Seed $E_0$ ($E^*$) | {g_headers} |\n")
                    f.write(f"|:---------:|:------------------:| {g_aligns} |\n")
                    for rank in sorted(rank_to_g.keys()):
                        g_map = rank_to_g[rank]
                        sample_tr = next(iter(g_map.values()))
                        e0 = sample_tr.get("SeedEnergy", "")
                        opt_e = sample_tr.get("KnownGlobalOpt", "")
                        pless_cells = " | ".join(f"{float(g_map[gv].get('StatevecWindowPless', 0.0)):.4f}" if gv in g_map else "N/A" for gv in g_vals)
                        f.write(f"| Seed #{rank} | **$E_0={e0}$** ($E^*={opt_e}$) | {pless_cells} |\n")
                    f.write("\n")

                    # Table 3: Mean <dH> vs g
                    f.write("### Quantum Dynamics Diffusion Distance $\\langle d_H \\rangle$ (Mean Bit Flips) vs. Transverse Field $g$\n\n")
                    f.write(f"| Seed Rank | Seed $E_0$ ($E^*$) | {g_headers} |\n")
                    f.write(f"|:---------:|:------------------:| {g_aligns} |\n")
                    for rank in sorted(rank_to_g.keys()):
                        g_map = rank_to_g[rank]
                        sample_tr = next(iter(g_map.values()))
                        e0 = sample_tr.get("SeedEnergy", "")
                        opt_e = sample_tr.get("KnownGlobalOpt", "")
                        flips_cells = " | ".join(f"{float(g_map[gv].get('MeanFlips', 0.0)):.1f}" if gv in g_map else "N/A" for gv in g_vals)
                        f.write(f"| Seed #{rank} | **$E_0={e0}$** ($E^*={opt_e}$) | {flips_cells} |\n")
                    f.write("\n")
        elif is_per_trap_format:
            f.write("# SKQD Benchmark Results\n\n")
            for (n, g, k, m, dt), trap_rows in scans.items():
                f.write(f"## System Size $N = {n}$, Field $g = {g:.2f}$ (Trotter: $K={k}, M={m}, \\Delta t={dt:.4f}$)\n\n")
                if has_standard_rates:
                    f.write("| Trap Level | Seed $E_0$ ($E^*$) | Algorithm | Min $E$ Found | Global Opt Hit % | Success Rate % ($E < E_0$) | Per-Eval Rate $r$ ($ev^{-1}$) | Classical Evals |\n")
                    f.write("|:----------:|:------------------:|:----------|:-------------:|:-----------------:|:--------------------------:|:-----------------------------:|:---------------:|\n")
                else:
                    f.write("| Trap Level | Seed $E_0$ ($E^*$) | Algorithm | Min $E$ Found | Global Opt Hit % | Success Rate % ($E < E_0$) | Mean Drop $\\Delta E$ | Classical Evals |\n")
                    f.write("|:----------:|:------------------:|:----------|:-------------:|:-----------------:|:--------------------------:|:--------------------:|:---------------:|\n")

                for tr in trap_rows:
                    rank = tr.get("TrapRank", "")
                    e0 = tr.get("SeedEnergy", "")
                    opt_e = tr.get("KnownGlobalOpt", "")
                    flips = float(tr.get("MeanFlips", 4.0))
                    kick_d = max(1, int(round(flips)))
                    seed_str = f"**$E_0={e0}$** ($E^*={opt_e}$)"

                    algos = [
                        ("1. SKQD Dynamics (Raw Subspace)", "SKQD"),
                        ("2. Dynamics + SD Quench", "Quench"),
                        (f"3. Random Kick (${kick_d}$ flips) + SD", "Kick"),
                        ("4. Multi-Start SD (Random Restarts)", "Restart"),
                        ("5. Blind Random Sampling", "Blind")
                    ]

                    for idx, (display_name, prefix) in enumerate(algos):
                        rank_col = f"**Level #{rank}**" if idx == 0 else ""
                        seed_col = seed_str if idx == 0 else ""
                        min_e = tr.get(f"{prefix}_MinE", "")
                        opt_pct = f"{float(tr.get(f'{prefix}_OptPct', 0.0)):.1f}%"
                        succ_pct = f"{float(tr.get(f'{prefix}_SuccPct', 0.0)):.1f}%"
                        drop = f"{float(tr.get(f'{prefix}_MeanDrop', 0.0)):+.2f}"
                        evals = f"{float(tr.get(f'{prefix}_MeanEvals', 0.0)):.0f}"
                        val_col = f"{float(tr.get(f'{prefix}_PerEvalRate', 0.0)):.2e}" if has_standard_rates else drop

                        f.write(f"| {rank_col} | {seed_col} | {display_name} | {min_e} | {opt_pct} | {succ_pct} | {val_col} | {evals} |\n")

                f.write("\n")
        else:
            f.write("# Benchmark Results\n\n")
            f.write("| $N$ | $g$ | $K$ | $M$ | $\\Delta t$ | Min Energy Found | Known $E^*$ | Global Opt Hit % | Success Rate % |\n")
            f.write("|:---:|:---:|:---:|:---:|:----------:|:----------------:|:-----------:|:-----------------:|:--------------:|\n")
            for r in rows:
                f.write(f"| {r.get('N')} | {float(r.get('g', 2.0)):.2f} | {r.get('K')} | {r.get('M')} | {float(r.get('dt', 0.0)):.4f} | {r.get('BestEnergyFound')} | {r.get('KnownGlobalOpt')} | {float(r.get('GlobalOptRate_Pct', 0.0)):.1f}% | {float(r.get('SuccessRate_Pct', 0.0)):.1f}% |\n")
            f.write("\n")

    print(f"[SUCCESS] Report written to '{output_md_path}'. Ready to share or send!")
    if output_md_path.endswith(".MD"):
        try:
            alt_path = output_md_path[:-3] + ".md"
            if not os.path.exists(alt_path):
                import shutil
                shutil.copyfile(output_md_path, alt_path)
        except Exception:
            pass
    elif output_md_path.endswith(".md"):
        try:
            alt_path = output_md_path[:-3] + ".MD"
            if not os.path.exists(alt_path):
                import shutil
                shutil.copyfile(output_md_path, alt_path)
        except Exception:
            pass
    return True

if __name__ == "__main__":
    summary_path = sys.argv[1] if len(sys.argv) > 1 else "skqd_summary_statistics.csv"
    if not os.path.exists(summary_path) and os.path.exists(os.path.join("build", summary_path)):
        summary_path = os.path.join("build", summary_path)

    dir_name = os.path.dirname(summary_path)
    base_name = os.path.basename(summary_path)

    if len(sys.argv) > 2:
        detailed_path = sys.argv[2]
    else:
        cand = ""
        if "_summary.csv" in base_name:
            cand = os.path.join(dir_name, base_name.replace("_summary.csv", "_results.csv"))
        elif "summary_statistics.csv" in base_name:
            cand = os.path.join(dir_name, base_name.replace("summary_statistics.csv", "benchmark_results.csv"))
        else:
            cand = os.path.join(dir_name, "skqd_benchmark_results.csv")
        detailed_path = cand if os.path.exists(cand) else ""

    if len(sys.argv) > 3:
        output_md_path = sys.argv[3]
    else:
        if "_summary.csv" in base_name:
            output_md_path = base_name.replace("_summary.csv", "_REPORT.md").upper()
        else:
            output_md_path = "SKQD_BENCHMARK_REPORT.md"

    generate_report(summary_path, detailed_path, output_md_path)
