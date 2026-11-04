#!/usr/bin/env python3
"""
Publication-Quality Figure and Table Generation Suite for Bernasconi-Krylov Benchmarks.
Strictly file-based: loads persistent structured JSON / CSV records directly from disk,
computes bootstrap error bars, and exports publication vector (.pdf) and raster (.png) figures
along with modular LaTeX table fragments (tables/tab_*.tex) directly inputted by the manuscript.

Figures Generated:
  (a) fig_a_ratio_R_vs_N: R = (window P_<) / r_classical* vs N for Level 1 and Level 2,
      showing sample median and B=2000 bootstrap confidence intervals.
  (b) fig_b_rise_rate_and_pless_vs_dmin: Initial rise rate dP_</dt and window P_< vs d_min.
  (c) fig_c_pless_and_dh_vs_g: Window P_< and diffusion <dH> vs g for N = 22 to 30.
  (d) fig_d_hardest_seed_dynamics_across_N: Continuous P_<(t) dynamics curves for hardest seed at each N.

Tables Exported:
  - tables/tab_seed_levels.tex           (Table 1: Exhaustive seed levels and hit fractions)
  - tables/tab_ratio_summary.tex         (Table 3: Quantum-to-classical advantage ratio summary)
  - tables/tab_hardest_seeds.tex         (Table 5: Hardest seed parameters and escape rates)
  - tables/tab_coupling_crossover.tex     (Table 6: Coupling crossover dynamics)
  - tables/tab_window_underestimation.tex (Table 7: Evaluation window underestimation factors)
  Also exports plot_data_summary.json with all exact plotted numbers.
"""

import sys
import os
import glob
import json
import argparse
import numpy as np

# Ensure local dependencies are accessible
lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib_py")
if os.path.exists(lib_dir) and lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)

# Set writable Matplotlib config directory
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_mpl")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

if HAS_MATPLOTLIB:
    # Publication Style configuration
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 1.8,
        "lines.markersize": 7,
        "figure.autolayout": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--"
    })

def bootstrap_median_ci(data: list, num_resamples: int = 2000, ci_level: float = 0.68):
    """Computes median and bootstrap confidence interval [ci_low, ci_high]."""
    if len(data) == 0:
        return 0.0, 0.0, 0.0
    arr = np.array(data, dtype=float)
    if len(arr) == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])

    med = float(np.median(arr))
    rng = np.random.default_rng(seed=42)
    boot_indices = rng.integers(0, len(arr), size=(num_resamples, len(arr)))
    boot_medians = np.median(arr[boot_indices], axis=1)

    alpha = (1.0 - ci_level) / 2.0
    ci_low = float(np.percentile(boot_medians, 100.0 * alpha))
    ci_high = float(np.percentile(boot_medians, 100.0 * (1.0 - alpha)))
    return med, ci_low, ci_high

def get_known_optimum(n_spins: int):
    opt_map = {16: 24, 20: 26, 22: 39, 24: 36, 26: 45, 28: 50, 30: 59}
    return opt_map.get(n_spins, None)

def load_exhaustive_minima(data_dir: str):
    """Loads exhaustive local minima search results for N in [20, 22, 24, 26, 28, 30]."""
    exhaustive_data = {}
    candidate_dirs = [data_dir, "data_processed", "../data_processed", "data_raw", "../data_raw"]
    
    for n in [20, 22, 24, 26, 28, 30]:
        found = False
        for d in candidate_dirs:
            p = os.path.join(d, f"exhaustive_minima_n{n}.json")
            if os.path.exists(p):
                with open(p, "r") as fp:
                    exhaustive_data[n] = json.load(fp)
                found = True
                break
        if not found:
            print(f"[Notice] exhaustive_minima_n{n}.json not found in candidate paths.")
    return exhaustive_data

def load_all_structured_data(data_dir: str, min_n: int = 20):
    """Loads all structured benchmark JSON files from data_dir, excluding global optimum states and N < min_n."""
    files = sorted(glob.glob(os.path.join(data_dir, "structured_benchmark_n*.json")))
    if not files:
        alt_dir = os.path.join(data_dir, "data_processed")
        if os.path.exists(alt_dir):
            files = sorted(glob.glob(os.path.join(alt_dir, "structured_benchmark_n*.json")))

    datasets = {}
    for f in files:
        with open(f, "r") as fp:
            d = json.load(fp)
            n = d.get("system", {}).get("N")
            if n and n >= min_n:
                opt_e = get_known_optimum(n)
                if opt_e is not None and "seeds" in d:
                    d["seeds"] = [s for s in d["seeds"] if s.get("trap_energy_E0", 999999) > opt_e]
                    d["system"]["num_seeds"] = len(d["seeds"])
                datasets[n] = d
    return datasets

# =============================================================================
# TABLE 1: tab_seed_levels.tex
# =============================================================================
def export_table_1_seed_levels(exhaustive_data: dict, tables_dir: str):
    """
    Exports Table 1: Exhaustive seed levels and hit fractions for N in [20, 22, 24, 26, 28, 30].
    Columns: $N$ & $E^*$ & Tier & $E_0$ & Orbits & $d_{\\min}$ range & SD hit fraction
    """
    os.makedirs(tables_dir, exist_ok=True)
    out_path = os.path.join(tables_dir, "tab_seed_levels.tex")

    lines = [
        r"\begin{tabular}{ccccccc}",
        r"\toprule",
        r"$N$ & $E^*$ & Tier & $E_0$ & Orbits & $d_{\min}$ range & SD hit fraction \\",
        r"\midrule"
    ]

    for n in sorted(exhaustive_data.keys()):
        ed = exhaustive_data[n]
        e_star = ed.get("E_star", get_known_optimum(n))
        levels = ed.get("levels", [])

        for idx, lvl in enumerate(levels[:2]):
            tier_name = f"Level {idx + 1}"
            e0 = lvl.get("E0", 0)
            orbits = lvl.get("orbits", [])
            num_orbits = len(orbits)
            
            dmins = [o.get("d_min", 0) for o in orbits if "d_min" in o]
            if dmins:
                d_min_str = f"{min(dmins)}--{max(dmins)}" if min(dmins) != max(dmins) else f"{min(dmins)}"
            else:
                d_min_str = "---"

            tot_sd_hit = sum(o.get("sd_hit_fraction", 0.0) for o in orbits)
            sd_str = f"{tot_sd_hit * 100.0:.2f}\\%"

            n_col = f"{n}" if idx == 0 else ""
            e_star_col = f"{e_star}" if idx == 0 else ""

            lines.append(f"{n_col:2} & {e_star_col:2} & {tier_name} & {e0:2} & {num_orbits:3} & {d_min_str:5} & {sd_str} \\\\")

        if n != sorted(exhaustive_data.keys())[-1]:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}"
    ])

    with open(out_path, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"[plot_paper_figures] Exported Table 1 to: {out_path}")

# =============================================================================
# FIGURE (a): Ratio R = window P_< / r_classical* vs N
# =============================================================================
def generate_figure_a(datasets: dict, exhaustive_data: dict, output_dir: str,
                      num_resamples: int = 2000, target_g: float = 0.35):
    """
    Figure (a): Quantum advantage ratio R = <P_<> / r_classical* vs N at field g.
    Where r_classical* = max(best_d_rate, restart_rate, tabu_best_rate).
    If classical rate is 0, uses one-hit lower bound R >= win_p * n_eval.
    Single-orbit levels report n=1 without bootstrapping.
    """
    fig_data = {
        "target_g": target_g,
        "Level_1": {"N": [], "E0": [], "n_orbits": [], "d_min_range": [], "median": [], "median_R": [], "ci": [], "ci_low": [], "ci_high": [],
                    "error_low": [], "error_high": [], "is_lower_bound": [], "lower_bound_flags": [], "raw_ratios": [], "per_orbit_records": []},
        "Level_2": {"N": [], "E0": [], "n_orbits": [], "d_min_range": [], "median": [], "median_R": [], "ci": [], "ci_low": [], "ci_high": [],
                    "error_low": [], "error_high": [], "is_lower_bound": [], "lower_bound_flags": [], "raw_ratios": [], "per_orbit_records": []}
    }

    g_keys = [f"{target_g:.2f}", f"{target_g:.1f}", str(target_g)]
    n_all = [20, 22, 24, 26, 28, 30]

    for n in n_all:
        if n not in datasets:
            continue
        data = datasets[n]
        seeds = data.get("seeds", [])
        if not seeds:
            continue

        ex_levels = exhaustive_data.get(n, {}).get("levels", [])
        if len(ex_levels) >= 1:
            lvl_configs = [("Level_1", ex_levels[0]["E0"]), ("Level_2", ex_levels[1]["E0"] if len(ex_levels) > 1 else None)]
        else:
            # Fallback to unique energies
            energy_groups = {}
            for s in seeds:
                energy_groups.setdefault(s["trap_energy_E0"], []).append(s)
            s_energies = sorted(energy_groups.keys())
            lvl_configs = [("Level_1", s_energies[0])]
            if len(s_energies) > 1:
                lvl_configs.append(("Level_2", s_energies[1]))

        for lvl_name, e_target in lvl_configs:
            if e_target is None:
                continue

            lvl_seeds = [s for s in seeds if s.get("trap_energy_E0") == e_target]
            if not lvl_seeds:
                continue

            ratios = []
            is_lower_bound = []
            orbit_records = []

            for s in lvl_seeds:
                cb = s.get("classical_baselines", {})
                c_rate = max(
                    cb.get("best_d_rate", 0.0),
                    cb.get("restart_rate", 0.0),
                    cb.get("tabu_best_rate", 0.0)
                )

                q_dyn = s.get("quantum_dynamics", {})
                q_run = None
                for gk in g_keys:
                    if gk in q_dyn:
                        q_run = q_dyn[gk]
                        break

                if q_run is None:
                    continue

                win_p = q_run.get("window_avg_p_less", 0.0)
                sd_rate = max(cb.get("best_d_rate", 0.0), cb.get("restart_rate", 0.0))
                tb_rate = cb.get("tabu_best_rate", 0.0)

                if c_rate > 0.0:
                    r = win_p / c_rate
                    ratios.append(r)
                    is_lower_bound.append(False)
                else:
                    n_eval = max(
                        cb.get("total_evaluations", 0),
                        cb.get("tabu_total_evaluations", 0),
                        cb.get("best_d_evals", 0) + cb.get("restart_evals", 0),
                        100000
                    )
                    r = win_p * n_eval
                    ratios.append(r)
                    is_lower_bound.append(True)

                orbit_records.append({
                    "canonical_id": s.get("canonical_id", ""),
                    "d_min": s.get("d_min", 0),
                    "win_p": win_p,
                    "sd_best_rate": sd_rate,
                    "tabu_best_rate": tb_rate,
                    "c_rate": c_rate,
                    "ratio": r,
                    "is_lower_bound": (c_rate <= 0.0)
                })

            if ratios:
                n_orb = len(ratios)
                if n_orb == 1:
                    med = float(ratios[0])
                    ci_l, ci_h = med, med
                    err_l, err_h = 0.0, 0.0
                else:
                    med, ci_l, ci_h = bootstrap_median_ci(ratios, num_resamples=num_resamples)
                    err_l = max(1e-5, med - ci_l)
                    err_h = max(1e-5, ci_h - med)

                has_lb = any(is_lower_bound)
                d_mins = [o["d_min"] for o in orbit_records if o["d_min"] > 0]
                dm_range = [min(d_mins), max(d_mins)] if d_mins else []

                fig_data[lvl_name]["N"].append(n)
                fig_data[lvl_name]["E0"].append(e_target)
                fig_data[lvl_name]["n_orbits"].append(n_orb)
                fig_data[lvl_name]["d_min_range"].append(dm_range)
                fig_data[lvl_name]["median"].append(med)
                fig_data[lvl_name]["median_R"].append(med)
                fig_data[lvl_name]["ci"].append([ci_l, ci_h])
                fig_data[lvl_name]["ci_low"].append(ci_l)
                fig_data[lvl_name]["ci_high"].append(ci_h)
                fig_data[lvl_name]["error_low"].append(err_l)
                fig_data[lvl_name]["error_high"].append(err_h)
                fig_data[lvl_name]["is_lower_bound"].append(has_lb)
                fig_data[lvl_name]["lower_bound_flags"].append(list(is_lower_bound))
                fig_data[lvl_name]["raw_ratios"].append(ratios)
                fig_data[lvl_name]["per_orbit_records"].append(orbit_records)

    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=300)

        # Plot Level 1
        l1 = fig_data["Level_1"]
        if l1["N"]:
            for i, (n, med, el, eh, lb) in enumerate(zip(l1["N"], l1["median"], l1["error_low"], l1["error_high"], l1["is_lower_bound"])):
                marker = "^" if lb else "o"
                ax.errorbar(
                    [n], [med], yerr=[[el], [eh]],
                    fmt=marker, color="#1f77b4", ecolor="#1f77b4", elinewidth=2, capsize=4, capthick=1.5,
                    label="Hardest Level 1 (Deepest Traps $E_0^*$)" if i == 0 else ""
                )
            ax.plot(l1["N"], l1["median"], "-", color="#1f77b4", linewidth=1.5, zorder=3)

        # Plot Level 2
        l2 = fig_data["Level_2"]
        if l2["N"]:
            for i, (n, med, el, eh, lb) in enumerate(zip(l2["N"], l2["median"], l2["error_low"], l2["error_high"], l2["is_lower_bound"])):
                marker = "^" if lb else "s"
                ax.errorbar(
                    [n + 0.18], [med], yerr=[[el], [eh]],
                    fmt=marker, color="#d62728", ecolor="#d62728", elinewidth=2, capsize=4, capthick=1.5,
                    label="Hardest Level 2 ($E_0 > E_0^*$)" if i == 0 else ""
                )
            ax.plot(np.array(l2["N"]) + 0.18, l2["median"], "--", color="#d62728", linewidth=1.5, zorder=3)

        ax.axhline(1.0, color="#555555", linestyle=":", linewidth=1.5, label="Parity ($R = 1$)", zorder=2)
        ax.set_yscale("log")
        ax.set_xlabel("System Size $N$")
        ax.set_ylabel(r"Quantum Advantage Ratio $R = \langle P_< \rangle / r_{\mathrm{classical}}^*$")
        ax.set_title(rf"$\mathbf{{(a)}}$ Ratio $R$ vs $N$ for Hardest Two Minima Levels ($g = {target_g:.2f}$)", pad=10)
        ax.set_xticks(n_all)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(frameon=True, loc="upper right")

        png_path = os.path.join(output_dir, "fig_a_ratio_R_vs_N.png")
        pdf_path = os.path.join(output_dir, "fig_a_ratio_R_vs_N.pdf")
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_paper_figures] Saved Figure (a) to: {png_path} and {pdf_path}")

    return fig_data

# =============================================================================
# TABLE 3: tab_ratio_summary.tex
# =============================================================================
def export_table_3_ratio_summary(fig_a_data: dict, exhaustive_data: dict, tables_dir: str):
    """
    Exports Table 3: Summary of quantum-to-classical advantage ratio R.
    Columns: N = 20, 22, 24, 26, 28, 30.
    """
    os.makedirs(tables_dir, exist_ok=True)
    out_path = os.path.join(tables_dir, "tab_ratio_summary.tex")

    n_cols = [20, 22, 24, 26, 28, 30]
    header = " & ".join([f"$N = {n}$" for n in n_cols])

    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        rf"\textbf{{Energy Tier}} & {header} \\",
        r"\midrule",
        r"\textbf{Level 1 (Deepest Traps)} & & & & & & \\"
    ]

    l1 = fig_a_data["Level_1"]
    l1_map = {n: i for i, n in enumerate(l1["N"])}

    # Level 1 rows: E0, Orbits, dmin, Median R, 68% CI
    e0_row = ["Trap Energy $E_0$"]
    orb_row = [r"Distinct Orbits ($n$)"]
    dmin_row = [r"Barrier Distance $d_{\min}$ Range"]
    med_row = [r"Sample Median $R$"]
    ci_row = [r"$68\%$ Bootstrap CI"]

    for n in n_cols:
        ex_lvl1 = exhaustive_data.get(n, {}).get("levels", [{}])[0]
        e0 = ex_lvl1.get("E0", "---")
        orbits = ex_lvl1.get("orbits", [])
        dmins = [o.get("d_min", 0) for o in orbits if "d_min" in o]
        dm_str = f"{min(dmins)}--{max(dmins)}" if dmins and min(dmins) != max(dmins) else (f"{min(dmins)}" if dmins else "---")

        e0_row.append(str(e0))
        orb_row.append(str(len(orbits)) if orbits else "---")
        dmin_row.append(dm_str)

        if n in l1_map:
            idx = l1_map[n]
            med = l1["median"][idx]
            lb = l1["is_lower_bound"][idx]
            prefix = r"\ge " if lb else ""
            med_row.append(f"{prefix}\\mathbf{{{med:.2f}}}")

            if l1["n_orbits"][idx] == 1:
                ci_row.append("$n=1$")
            else:
                ci_row.append(f"[{l1['ci_low'][idx]:.2f}, {l1['ci_high'][idx]:.2f}]")
        else:
            med_row.append("---")
            ci_row.append("---")

    lines.append(" & ".join(e0_row) + r" \\")
    lines.append(" & ".join(orb_row) + r" \\")
    lines.append(" & ".join(dmin_row) + r" \\")
    lines.append(" & ".join(med_row) + r" \\")
    lines.append(" & ".join(ci_row) + r" \\")

    lines.extend([
        r"\midrule",
        r"\textbf{Level 2 (Excited Traps)} & & & & & & \\"
    ])

    l2 = fig_a_data["Level_2"]
    l2_map = {n: i for i, n in enumerate(l2["N"])}

    e0_row2 = ["Trap Energy $E_0$"]
    orb_row2 = [r"Distinct Orbits ($n$)"]
    dmin_row2 = [r"Barrier Distance $d_{\min}$ Range"]
    med_row2 = [r"Sample Median $R$"]
    ci_row2 = [r"$68\%$ Bootstrap CI"]

    for n in n_cols:
        ex_lvl2 = exhaustive_data.get(n, {}).get("levels", [{}, {}])[1] if len(exhaustive_data.get(n, {}).get("levels", [])) > 1 else {}
        e0 = ex_lvl2.get("E0", "---")
        orbits = ex_lvl2.get("orbits", [])
        dmins = [o.get("d_min", 0) for o in orbits if "d_min" in o]
        dm_str = f"{min(dmins)}--{max(dmins)}" if dmins and min(dmins) != max(dmins) else (f"{min(dmins)}" if dmins else "---")

        e0_row2.append(str(e0))
        orb_row2.append(str(len(orbits)) if orbits else "---")
        dmin_row2.append(dm_str)

        if n in l2_map:
            idx = l2_map[n]
            med = l2["median"][idx]
            lb = l2["is_lower_bound"][idx]
            prefix = r"\ge " if lb else ""
            med_row2.append(f"{prefix}\\mathbf{{{med:.2f}}}")

            if l2["n_orbits"][idx] == 1:
                ci_row2.append("$n=1$")
            else:
                ci_row2.append(f"[{l2['ci_low'][idx]:.2f}, {l2['ci_high'][idx]:.2f}]")
        else:
            med_row2.append("---")
            ci_row2.append("---")

    lines.append(" & ".join(e0_row2) + r" \\")
    lines.append(" & ".join(orb_row2) + r" \\")
    lines.append(" & ".join(dmin_row2) + r" \\")
    lines.append(" & ".join(med_row2) + r" \\")
    lines.append(" & ".join(ci_row2) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}"
    ])

    with open(out_path, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"[plot_paper_figures] Exported Table 3 to: {out_path}")

# =============================================================================
# TABLE 5: tab_hardest_seeds.tex
# =============================================================================
def export_table_5_hardest_seeds(datasets: dict, tables_dir: str, target_g: float = 0.35):
    """
    Exports Table 5: Dynamical parameters and escape efficiency for hardest local minimum at each N.
    Columns: $N$ & $E_0$ & $d_{\min}$ & $P_{\mathrm{blind}}$ & Window $\langle P_< \rangle$ & Best Classical Method & $r^*_{\mathrm{classical}}$ & Ratio $R$
    """
    os.makedirs(tables_dir, exist_ok=True)
    out_path = os.path.join(tables_dir, "tab_hardest_seeds.tex")

    lines = [
        r"\begin{tabular}{cccccccc}",
        r"\toprule",
        r"$N$ & $E_0$ & $d_{\min}$ & $P_{\mathrm{blind}}$ & Window $\langle P_< \rangle$ & Best Classical Method & $r_{\mathrm{classical}}^*$ & Ratio $R$ \\",
        r"\midrule"
    ]

    g_keys = [f"{target_g:.2f}", f"{target_g:.1f}", str(target_g)]

    for n in sorted(datasets.keys()):
        data = datasets[n]
        seeds = data.get("seeds", [])
        if not seeds:
            continue

        deepest = min(seeds, key=lambda s: (s.get("trap_energy_E0", 9999), -s.get("d_min", 0)))
        e0 = deepest.get("trap_energy_E0", 0)
        dmin = deepest.get("d_min", 0)

        q_dyn = deepest.get("quantum_dynamics", {})
        q_run = q_dyn.get("0.35")
        if not q_run:
            continue

        win_p = q_run.get("window_avg_p_less", 0.0) if q_run else 0.0
        p_blind = q_run.get("p_blind", 0.0) if q_run else 0.0

        cb = deepest.get("classical_baselines", {})
        sd_rate = cb.get("best_d_rate", 0.0)
        rst_rate = cb.get("restart_rate", 0.0)
        tb_rate = cb.get("tabu_best_rate", 0.0)

        c_rate = max(sd_rate, rst_rate, tb_rate)

        # Identify best classical method
        if tb_rate >= max(sd_rate, rst_rate) and tb_rate > 0:
            method_str = cb.get("tabu_best_method", "Tabu Search")
        elif sd_rate >= rst_rate and sd_rate > 0:
            best_d = cb.get("best_d", 0)
            method_str = f"Kick-SD ($d={best_d}$)"
        elif rst_rate > 0:
            method_str = "Restart-SD"
        else:
            method_str = "Classical Upper Bound"

        if c_rate > 0:
            r = win_p / c_rate
            r_str = f"\\mathbf{{{r:.1f}\\times}}"
            crate_str = f"{c_rate:.2e}"
        else:
            n_eval = max(cb.get("total_evaluations", 0), cb.get("tabu_total_evaluations", 0), 100000)
            r = win_p * n_eval
            r_str = f"\\mathbf{{\\ge {r:.1f}\\times}}"
            crate_str = f"< 1/{n_eval}"

        lines.append(f"{n:2} & {e0:2} & {dmin:2} & {p_blind:.2e} & {win_p:.2e} & {method_str} & {crate_str} & {r_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}"
    ])

    with open(out_path, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"[plot_paper_figures] Exported Table 5 to: {out_path}")

# =============================================================================
# TABLE 6: tab_coupling_crossover.tex
# =============================================================================
def export_table_6_coupling_crossover(datasets: dict, tables_dir: str):
    """
    Exports Table 6: Escape probability P_<(t) comparing couplings g = 0.30 and g = 0.35.
    """
    os.makedirs(tables_dir, exist_ok=True)
    out_path = os.path.join(tables_dir, "tab_coupling_crossover.tex")

    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"$N$ & Time $t$ & $P_<(0.30)$ & $P_<(0.35)$ & $\frac{P_<(0.30)}{P_<(0.35)}$ & Dynamic Regime \\",
        r"\midrule"
    ]

    selected_cases = [
        (24, 1, [5.0, 10.0, 38.5, 50.0, 55.0]),
        (28, 2, [10.0, 38.5, 60.0]),
        (30, 2, [10.0, 38.5, 60.0])
    ]

    for c_idx, (n, rank_target, eval_times) in enumerate(selected_cases):
        data = datasets.get(n, {})
        seeds = data.get("seeds", [])
        if not seeds:
            continue

        # Find target seed
        seed = next((s for s in seeds if s.get("trap_rank") == rank_target), seeds[0])
        q_dyn = seed.get("quantum_dynamics", {})
        q_30 = q_dyn.get("0.30") or q_dyn.get("0.3")
        q_35 = q_dyn.get("0.35")

        if not q_30 or not q_35:
            continue

        t_arr30 = np.array(q_30.get("time_array", []))
        p_arr30 = np.array(q_30.get("p_less_array", []))
        t_arr35 = np.array(q_35.get("time_array", []))
        p_arr35 = np.array(q_35.get("p_less_array", []))

        lines.append(f"\\textbf{{$N = {n}$ (Rank {rank_target})}} & & & & & \\\\")

        for t_val in eval_times:
            if len(t_arr30) == 0 or len(t_arr35) == 0:
                continue
            idx30 = int(np.argmin(np.abs(t_arr30 - t_val)))
            idx35 = int(np.argmin(np.abs(t_arr35 - t_val)))
            p30 = p_arr30[idx30]
            p35 = p_arr35[idx35]
            ratio = (p30 / p35) if p35 > 0 else 1.0

            if ratio < 0.6:
                regime = "$g=0.35$ faster rise"
            elif ratio < 0.95:
                regime = "$g=0.30$ catching up"
            elif ratio <= 1.15:
                regime = "\\textbf{Crossover}"
            else:
                regime = "\\textbf{$g=0.30$ superior}"

            ratio_str = f"\\mathbf{{{ratio:.2f}}}" if ratio >= 1.0 else f"{ratio:.2f}"
            lines.append(f"& {t_val:.1f} & {p30:.2e} & {p35:.2e} & {ratio_str} & {regime} \\\\")

        if c_idx < len(selected_cases) - 1:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}"
    ])

    with open(out_path, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"[plot_paper_figures] Exported Table 6 to: {out_path}")

# =============================================================================
# TABLE 7: tab_window_underestimation.tex
# =============================================================================
def export_table_7_window_underestimation(datasets: dict, tables_dir: str):
    """
    Exports Table 7: Comparison of window-averaged P_< versus late-time P_<(t=60).
    """
    os.makedirs(tables_dir, exist_ok=True)
    out_path = os.path.join(tables_dir, "tab_window_underestimation.tex")

    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"$N$ & Trap $E_0$ & $d_{\min}$ & Field $g$ & $\langle P_< \rangle_{[10, 38.5]}$ & $P_<(t = 60)$ & Factor \\",
        r"\midrule"
    ]

    selected_rows = [
        (24, 0.30, 1),
        (24, 0.30, 2),
        (28, 0.35, 1),
        (28, 0.30, 2),
        (30, 0.35, 1),
        (30, 0.30, 2)
    ]

    for n, g_val, rank_target in selected_rows:
        data = datasets.get(n, {})
        seeds = data.get("seeds", [])
        if not seeds:
            continue

        seed = next((s for s in seeds if s.get("trap_rank") == rank_target), seeds[0])
        e0 = seed.get("trap_energy_E0", 0)
        dmin = seed.get("d_min", 0)

        g_key = f"{g_val:.2f}"
        q_run = seed.get("quantum_dynamics", {}).get(g_key) or seed.get("quantum_dynamics", {}).get(f"{g_val:.1f}")

        if not q_run:
            continue

        win_p = q_run.get("window_avg_p_less", 0.0)
        p_arr = np.array(q_run.get("p_less_array", []))
        p60 = p_arr[-1] if len(p_arr) > 0 else win_p
        factor = (p60 / win_p) if win_p > 0 else 1.0

        lines.append(f"{n:2} & {e0:2} & {dmin:2} & {g_val:.2f} & {win_p:.2e} & {p60:.2e} & {factor:.2f}$\\times$ \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}"
    ])

    with open(out_path, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"[plot_paper_figures] Exported Table 7 to: {out_path}")

# =============================================================================
# FIGURE (b): Initial Rise Rate dP_</dt & Window P_< vs d_min
# =============================================================================
def generate_figure_b(datasets: dict, output_dir: str):
    fig_data = {"seeds": []}
    color_map = {20: "#2ca02c", 22: "#17becf", 24: "#ff7f0e", 26: "#9467bd", 28: "#d62728", 30: "#8c564b"}
    marker_map = {0.30: "^", 0.35: "o", 0.45: "s"}

    for n, data in sorted(datasets.items()):
        for s in data.get("seeds", []):
            dmin = s["d_min"]
            for g_str, q in s.get("quantum_dynamics", {}).items():
                try:
                    g_val = float(g_str)
                except ValueError:
                    continue
                rise_rate = q.get("initial_rise_rate_avg", 0.0)
                win_p = q.get("window_avg_p_less", 0.0)

                if rise_rate > 0 and win_p > 0:
                    fig_data["seeds"].append({
                        "N": n, "g": g_val, "d_min": dmin,
                        "rise_rate": rise_rate, "window_avg_p_less": win_p
                    })

    if HAS_MATPLOTLIB and fig_data["seeds"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0), dpi=300)
        for item in fig_data["seeds"]:
            c = color_map.get(item["N"], "#333333")
            m = marker_map.get(round(item["g"], 2), "o")
            ax1.scatter(item["d_min"], item["rise_rate"], color=c, marker=m, s=45, alpha=0.75, edgecolors="none")
            ax2.scatter(item["d_min"], item["window_avg_p_less"], color=c, marker=m, s=45, alpha=0.75, edgecolors="none")

        n_legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=f"$N={n}$")
                    for n, c in sorted(color_map.items()) if n in datasets]
        g_legend = [Line2D([0], [0], marker=m, color="k", linestyle="None", markersize=7, label=f"$g={g:.2f}$")
                    for g, m in marker_map.items()]

        ax1.set_yscale("log")
        ax1.set_xlabel(r"Minimal Barrier Distance $d_{\min}$")
        ax1.set_ylabel(r"Initial Rise Rate $\left.\frac{dP_<}{dt}\right|_{\mathrm{rise}}$ [$\mathrm{s}^{-1}$]")
        ax1.set_title(r"Initial Rise Rate vs $d_{\min}$", pad=10)
        ax1.grid(True, which="both", linestyle="--", alpha=0.3)
        ax1.legend(handles=n_legend, title="System Size", loc="upper right", framealpha=0.9)

        ax2.set_yscale("log")
        ax2.set_xlabel(r"Minimal Barrier Distance $d_{\min}$")
        ax2.set_ylabel(r"Window-Averaged Probability $\langle P_< \rangle$")
        ax2.set_title(r"Window $\langle P_< \rangle$ vs $d_{\min}$", pad=10)
        ax2.grid(True, which="both", linestyle="--", alpha=0.3)
        ax2.legend(handles=g_legend, title="Field $g$", loc="upper right", framealpha=0.9)

        fig.suptitle(r"$\mathbf{(b)}$ Quantum Escape Scaling with Barrier Distance $d_{\min}$", fontsize=14, y=1.02)
        png_path = os.path.join(output_dir, "fig_b_rise_rate_and_pless_vs_dmin.png")
        pdf_path = os.path.join(output_dir, "fig_b_rise_rate_and_pless_vs_dmin.pdf")
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_paper_figures] Saved Figure (b) to: {png_path} and {pdf_path}")

    return fig_data

# =============================================================================
# FIGURE (c): P_< and <dH> vs g at N = 22 to 30
# =============================================================================
def generate_figure_c(datasets: dict, output_dir: str):
    fig_data = {"N_curves": {}}
    color_map = {22: "#17becf", 24: "#1f77b4", 26: "#2ca02c", 28: "#d62728", 30: "#9467bd"}
    marker_map = {22: "v:", 24: "o-", 26: "s--", 28: "^-.", 30: "D:"}

    for n in [22, 24, 26, 28, 30]:
        if n not in datasets:
            continue
        data = datasets[n]
        seeds = data.get("seeds", [])
        if not seeds:
            continue

        g_pless, g_dh = {}, {}
        for s in seeds:
            for g_str, q in s.get("quantum_dynamics", {}).items():
                try:
                    g_val = float(g_str)
                except ValueError:
                    continue
                g_pless.setdefault(g_val, []).append(q.get("window_avg_p_less", 0.0))
                g_dh.setdefault(g_val, []).append(q.get("window_avg_dH", 0.0))

        if not g_pless:
            continue

        g_list = sorted(g_pless.keys())
        mean_p = [float(np.mean(g_pless[g])) for g in g_list]
        std_p = [float(np.std(g_pless[g]) / np.sqrt(len(g_pless[g]))) if len(g_pless[g]) > 1 else 0.0 for g in g_list]

        mean_dh = [float(np.mean(g_dh[g])) for g in g_list]
        std_dh = [float(np.std(g_dh[g]) / np.sqrt(len(g_dh[g]))) if len(g_dh[g]) > 1 else 0.0 for g in g_list]

        fig_data["N_curves"][n] = {
            "g": g_list,
            "mean_pless": mean_p, "sem_pless": std_p,
            "mean_dh": mean_dh, "sem_dh": std_dh
        }

    if HAS_MATPLOTLIB and fig_data["N_curves"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0), dpi=300)
        for n, c_data in fig_data["N_curves"].items():
            c = color_map.get(n, "#333333")
            fmt = marker_map.get(n, "o-")
            ax1.errorbar(c_data["g"], c_data["mean_pless"], yerr=c_data["sem_pless"], fmt=fmt, color=c,
                         elinewidth=1.6, capsize=3.5, label=f"$N = {n}$", zorder=4)
            ax2.errorbar(c_data["g"], c_data["mean_dh"], yerr=c_data["sem_dh"], fmt=fmt, color=c,
                         elinewidth=1.6, capsize=3.5, label=f"$N = {n}$", zorder=4)

        ax1.set_yscale("log")
        ax1.set_xlabel("Transverse Field Strength $g$")
        ax1.set_ylabel(r"Window-Averaged Probability $\langle P_< \rangle$")
        ax1.set_title(r"Tunneling Escape $\langle P_< \rangle$ vs $g$", pad=10)
        ax1.grid(True, which="both", linestyle="--", alpha=0.3)
        ax1.legend(frameon=True, loc="upper left")

        ax2.set_xlabel("Transverse Field Strength $g$")
        ax2.set_ylabel(r"Quantum Hamming Diffusion $\langle d_H \rangle$")
        ax2.set_title(r"Hamming Diffusion $\langle d_H \rangle$ vs $g$", pad=10)
        ax2.grid(True, linestyle="--", alpha=0.3)
        ax2.legend(frameon=True, loc="upper left")

        fig.suptitle(r"$\mathbf{(c)}$ Quantum Field Response across System Sizes ($N = 22 \text{ to } 30$)", fontsize=14, y=1.02)
        png_path = os.path.join(output_dir, "fig_c_pless_and_dh_vs_g.png")
        pdf_path = os.path.join(output_dir, "fig_c_pless_and_dh_vs_g.pdf")
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_paper_figures] Saved Figure (c) to: {png_path} and {pdf_path}")

    return fig_data

# =============================================================================
# FIGURE (d): P_<(t) Curves for Hardest Seed at Each N
# =============================================================================
def generate_figure_d(datasets: dict, output_dir: str):
    fig_data = {"curves": {}}
    color_map = {20: "#2ca02c", 22: "#17becf", 24: "#ff7f0e", 26: "#9467bd", 28: "#d62728", 30: "#8c564b"}
    target_g_key = "0.35"

    for n in sorted(datasets.keys()):
        data = datasets[n]
        seeds = data.get("seeds", [])
        if not seeds:
            continue

        deepest = min(seeds, key=lambda s: s["trap_energy_E0"])
        q_dyn = deepest.get("quantum_dynamics", {})
        q_run = q_dyn.get(target_g_key)
        if not q_run:
            continue

        t_arr = np.array(q_run.get("time_array", []))
        p_arr = np.array(q_run.get("p_less_array", []))
        p_blind = q_run.get("p_blind", 0.0)

        if len(t_arr) > 0 and len(p_arr) > 0:
            fig_data["curves"][n] = {
                "trap_rank": deepest.get("trap_rank", 1),
                "energy_E0": deepest["trap_energy_E0"],
                "d_min": deepest["d_min"],
                "p_blind": p_blind,
                "peak_p_less": q_run.get("peak_p_less", float(np.max(p_arr))),
                "window_avg_p_less": q_run.get("window_avg_p_less", float(np.mean(p_arr))),
                "time_array": t_arr.tolist(),
                "p_less_array": p_arr.tolist()
            }

    if HAS_MATPLOTLIB and fig_data["curves"]:
        fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
        for n, c_info in fig_data["curves"].items():
            c = color_map.get(n, "#333333")
            e0 = c_info["energy_E0"]
            dmin = c_info["d_min"]
            label_str = f"$N = {n}$ ($E_0^*={e0}, d_{{\\min}}={dmin}$)"
            ax.plot(c_info["time_array"], c_info["p_less_array"], label=label_str, color=c, alpha=0.9)
            if c_info["p_blind"] > 0:
                ax.axhline(c_info["p_blind"], color=c, linestyle=":", alpha=0.45, linewidth=1.0)

        ax.set_yscale("log")
        ax.set_xlim(0, 60.0)
        ax.set_xlabel("Evolution Time $t$ [$\hbar / J$]")
        ax.set_ylabel(r"Escape Probability $P_<(t)$")
        ax.set_title(r"$\mathbf{(d)}$ Continuous Probability Dynamics $P_<(t)$ for Deepest Trap ($g = 0.35$)", pad=10)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(frameon=True, loc="upper right")

        png_path = os.path.join(output_dir, "fig_d_hardest_seed_dynamics_across_N.png")
        pdf_path = os.path.join(output_dir, "fig_d_hardest_seed_dynamics_across_N.pdf")
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_paper_figures] Saved Figure (d) to: {png_path} and {pdf_path}")

    return fig_data

# =============================================================================
# CLI Main Driver
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Publication figure and table suite for Bernasconi-Krylov benchmarks.")
    parser.add_argument("--data-dir", default="data_processed", help="Path to structured benchmark JSON files")
    parser.add_argument("--output-dir", default="figures/paper", help="Output directory for generated figures")
    parser.add_argument("--tables-dir", default="tables", help="Output directory for generated LaTeX tables")
    parser.add_argument("--bootstrap-samples", type=int, default=2000, help="Number of bootstrap resamples (default: 2000)")
    parser.add_argument("--g-ratio", type=float, default=0.35, help="Transverse field coupling g for Figure (a) (default: 0.35)")
    parser.add_argument("--min-n", type=int, default=20, help="Minimum system size N to include (default: 20)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tables_dir, exist_ok=True)

    print(f"[plot_paper_figures] Loading exhaustive search results from: {args.data_dir}")
    exhaustive_data = load_exhaustive_minima(args.data_dir)

    # 1. Export Table 1 immediately from exhaustive minima results
    export_table_1_seed_levels(exhaustive_data, args.tables_dir)

    print(f"[plot_paper_figures] Loading persistent structured datasets from: {args.data_dir} (min_n = {args.min_n})")
    datasets = load_all_structured_data(args.data_dir, min_n=args.min_n)
    print(f"[plot_paper_figures] Loaded system sizes: N = {sorted(datasets.keys())}")

    # 2. Generate Figure (a) and Table 3
    summary = {}
    fig_a_data = generate_figure_a(datasets, exhaustive_data, args.output_dir,
                                   num_resamples=args.bootstrap_samples, target_g=args.g_ratio)
    summary["fig_a_ratio_vs_N"] = fig_a_data
    export_table_3_ratio_summary(fig_a_data, exhaustive_data, args.tables_dir)

    # 3. Export Tables 5, 6, 7
    export_table_5_hardest_seeds(datasets, args.tables_dir, target_g=args.g_ratio)
    export_table_6_coupling_crossover(datasets, args.tables_dir)
    export_table_7_window_underestimation(datasets, args.tables_dir)

    # 4. Generate Figures (b), (c), (d)
    summary["fig_b_rise_rate_and_pless_vs_dmin"] = generate_figure_b(datasets, args.output_dir)
    summary["fig_c_pless_and_dh_vs_g"] = generate_figure_c(datasets, args.output_dir)
    summary["fig_d_hardest_seed_dynamics"] = generate_figure_d(datasets, args.output_dir)

    # 5. Export JSON summary archive
    summary_path = os.path.join(args.output_dir, "plot_data_summary.json")
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"[plot_paper_figures] Exported exact plotted numbers archive to: {summary_path}")
    print("[plot_paper_figures] ALL FIGURES AND TABLES GENERATED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
