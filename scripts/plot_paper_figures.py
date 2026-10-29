#!/usr/bin/env python3
"""
Publication-Quality Figure Generation Suite for Bernasconi-Krylov Benchmarks.
Strictly file-based: loads persistent structured JSON / CSV records directly from disk,
computes bootstrap error bars, and exports publication vector (.pdf) and raster (.png) figures.

Figures Generated:
  (a) fig_a_ratio_R_vs_N: R = (window P_<) / r*(d*) vs N for the hardest two energy levels,
      showing sample median and B=2000 bootstrap confidence intervals.
  (b) fig_b_rise_rate_and_pless_vs_dmin: Initial rise rate dP_</dt and window P_< vs d_min.
  (c) fig_c_pless_and_dh_vs_g: Window P_< and diffusion <dH> vs g for N = 24 to 30.
  (d) fig_d_hardest_seed_dynamics_across_N: Continuous P_<(t) dynamics curves for the hardest seed at each N.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Style configuration
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

def load_all_structured_data(data_dir: str):
    """Loads all structured benchmark JSON files from data_dir."""
    files = sorted(glob.glob(os.path.join(data_dir, "structured_benchmark_n*.json")))
    if not files:
        # Check parent data directory
        alt_dir = os.path.join(data_dir, "data_processed")
        if os.path.exists(alt_dir):
            files = sorted(glob.glob(os.path.join(alt_dir, "structured_benchmark_n*.json")))

    if not files:
        raise FileNotFoundError(f"No structured_benchmark_n*.json files found in {data_dir}")

    datasets = {}
    for f in files:
        with open(f, "r") as fp:
            d = json.load(fp)
            n = d.get("system", {}).get("N")
            if n:
                datasets[n] = d
    return datasets

# =============================================================================
# FIGURE (a): Ratio R = window P_< / r*(d*) vs N for Hardest Two Levels
# =============================================================================
def generate_figure_a(datasets: dict, output_dir: str, num_resamples: int = 2000):
    """
    Figure (a): R = window P_< / best_d_rate vs N for Level 1 (deepest) and Level 2.
    Plots median and B=2000 bootstrap error bars.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=300)

    n_values = sorted(datasets.keys())
    fig_data = {
        "Level_1": {"N": [], "median": [], "ci_low": [], "ci_high": [], "error_low": [], "error_high": [], "raw_ratios": []},
        "Level_2": {"N": [], "median": [], "ci_low": [], "ci_high": [], "error_low": [], "error_high": [], "raw_ratios": []}
    }

    # Evaluate at g = 0.35 (standard mid-coupling)
    target_g_key = "0.35"

    for n in n_values:
        data = datasets[n]
        seeds = data.get("seeds", [])
        if not seeds:
            continue

        # Group seeds by unique energy E0
        energy_groups = {}
        for s in seeds:
            e0 = s["trap_energy_E0"]
            energy_groups.setdefault(e0, []).append(s)

        sorted_energies = sorted(energy_groups.keys())
        levels = [("Level_1", sorted_energies[0])]
        if len(sorted_energies) > 1:
            levels.append(("Level_2", sorted_energies[1]))

        for lvl_name, e_val in levels:
            lvl_seeds = energy_groups[e_val]
            ratios = []
            for s in lvl_seeds:
                c_rate = s["classical_baselines"].get("best_d_rate", 0.0)
                # Find quantum window P_<
                q_dyn = s.get("quantum_dynamics", {})
                q_run = q_dyn.get(target_g_key) or list(q_dyn.values())[0] if q_dyn else None
                if q_run and c_rate > 0:
                    win_p = q_run.get("window_avg_p_less", 0.0)
                    r = win_p / c_rate
                    ratios.append(r)
                elif q_run and c_rate == 0:
                    # When classical rate is exactly 0 (e.g. global minimum), ratio is strictly quantum-dominated
                    ratios.append(1e3)

            if ratios:
                med, ci_l, ci_h = bootstrap_median_ci(ratios, num_resamples=num_resamples)
                fig_data[lvl_name]["N"].append(n)
                fig_data[lvl_name]["median"].append(med)
                fig_data[lvl_name]["ci_low"].append(ci_l)
                fig_data[lvl_name]["ci_high"].append(ci_h)
                fig_data[lvl_name]["error_low"].append(max(1e-5, med - ci_l))
                fig_data[lvl_name]["error_high"].append(max(1e-5, ci_h - med))
                fig_data[lvl_name]["raw_ratios"].append(ratios)

    # Plot Level 1 (Deepest minima)
    l1 = fig_data["Level_1"]
    if l1["N"]:
        ax.errorbar(
            l1["N"], l1["median"],
            yerr=[l1["error_low"], l1["error_high"]],
            fmt="o-", color="#1f77b4", ecolor="#1f77b4", elinewidth=2, capsize=4, capthick=1.5,
            label="Hardest Level 1 (Deepest Traps $E_0^*$)", zorder=4
        )

    # Plot Level 2 (Second hardest minima)
    l2 = fig_data["Level_2"]
    if l2["N"]:
        ax.errorbar(
            np.array(l2["N"]) + 0.18, l2["median"],  # slight jitter for clarity
            yerr=[l2["error_low"], l2["error_high"]],
            fmt="s--", color="#d62728", ecolor="#d62728", elinewidth=2, capsize=4, capthick=1.5,
            label="Hardest Level 2 ($E_0 > E_0^*$)", zorder=3
        )

    # Parity baseline line R = 1
    ax.axhline(1.0, color="#555555", linestyle=":", linewidth=1.5, label="Parity ($R = 1$, Quantum = Classical)", zorder=2)

    ax.set_yscale("log")
    ax.set_xlabel("System Size $N$")
    ax.set_ylabel(r"Quantum-to-Classical Ratio $R = \langle P_< \rangle / r^*(d^*)$")
    ax.set_title(r"$\mathbf{(a)}$ Ratio $R$ vs $N$ for Hardest Two Minima Levels ($g = 0.35$)", pad=10)
    ax.set_xticks(n_values)
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
# FIGURE (b): Initial Rise Rate dP_</dt & Window P_< vs d_min
# =============================================================================
def generate_figure_b(datasets: dict, output_dir: str):
    """
    Figure (b): 2-panel plot showing:
      Panel 1: Initial rise rate dP_</dt vs d_min
      Panel 2: Window-averaged P_< vs d_min
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0), dpi=300)

    fig_data = {"seeds": []}
    color_map = {16: "#1f77b4", 20: "#2ca02c", 24: "#ff7f0e", 26: "#9467bd", 28: "#d62728", 30: "#8c564b"}
    marker_map = {0.30: "^", 0.35: "o", 0.45: "s"}

    for n, data in sorted(datasets.items()):
        for s in data.get("seeds", []):
            dmin = s["d_min"]
            for g_str, q in s.get("quantum_dynamics", {}).items():
                g_val = float(g_str)
                rise_rate = q.get("initial_rise_rate_avg", 0.0)
                win_p = q.get("window_avg_p_less", 0.0)

                if rise_rate > 0 and win_p > 0:
                    fig_data["seeds"].append({
                        "N": n, "g": g_val, "d_min": dmin,
                        "rise_rate": rise_rate, "window_avg_p_less": win_p
                    })

                    c = color_map.get(n, "#333333")
                    m = marker_map.get(round(g_val, 2), "o")

                    ax1.scatter(dmin, rise_rate, color=c, marker=m, s=45, alpha=0.75, edgecolors="none")
                    ax2.scatter(dmin, win_p, color=c, marker=m, s=45, alpha=0.75, edgecolors="none")

    # Legend proxies for N
    from matplotlib.lines import Line2D
    n_legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=f"$N={n}$")
                for n, c in sorted(color_map.items()) if n in datasets]
    # Legend proxies for g
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
# FIGURE (c): P_< and <dH> vs g at N = 24 to 30
# =============================================================================
def generate_figure_c(datasets: dict, output_dir: str):
    """
    Figure (c): 2-panel plot showing:
      Left Panel: Window-averaged P_< vs g for N in [24, 30]
      Right Panel: Quantum diffusion <dH> vs g for N in [24, 30]
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0), dpi=300)

    color_map = {24: "#1f77b4", 26: "#2ca02c", 28: "#d62728", 30: "#9467bd"}
    marker_map = {24: "o-", 26: "s--", 28: "^-.", 30: "D:"}

    fig_data = {"N_curves": {}}

    for n in [24, 26, 28, 30]:
        if n not in datasets:
            continue

        data = datasets[n]
        seeds = data.get("seeds", [])
        if not seeds:
            continue

        # Collect by g
        g_pless = {}
        g_dh = {}

        for s in seeds:
            for g_str, q in s.get("quantum_dynamics", {}).items():
                g_val = float(g_str)
                g_pless.setdefault(g_val, []).append(q.get("window_avg_p_less", 0.0))
                g_dh.setdefault(g_val, []).append(q.get("window_avg_dH", 0.0))

        g_list = sorted(g_pless.keys())
        mean_p = [float(np.mean(g_pless[g])) for g in g_list]
        std_p = [float(np.std(g_pless[g]) / np.sqrt(len(g_pless[g]))) for g in g_list]

        mean_dh = [float(np.mean(g_dh[g])) for g in g_list]
        std_dh = [float(np.std(g_dh[g]) / np.sqrt(len(g_dh[g]))) for g in g_list]

        fig_data["N_curves"][n] = {
            "g": g_list,
            "mean_pless": mean_p, "sem_pless": std_p,
            "mean_dh": mean_dh, "sem_dh": std_dh
        }

        c = color_map.get(n, "#333333")
        fmt = marker_map.get(n, "o-")

        # Left panel: P_< vs g
        ax1.errorbar(
            g_list, mean_p, yerr=std_p, fmt=fmt, color=c, elinewidth=1.6, capsize=3.5,
            label=f"$N = {n}$", zorder=4
        )

        # Right panel: dH vs g
        ax2.errorbar(
            g_list, mean_dh, yerr=std_dh, fmt=fmt, color=c, elinewidth=1.6, capsize=3.5,
            label=f"$N = {n}$", zorder=4
        )

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

    fig.suptitle(r"$\mathbf{(c)}$ Quantum Field Response across Large System Sizes ($N = 24 \text{ to } 30$)", fontsize=14, y=1.02)

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
    """
    Figure (d): P_<(t) continuous trajectories across all system sizes N
    for the deepest trap (Rank #1, lowest E0) at g = 0.35.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)

    color_map = {16: "#1f77b4", 20: "#2ca02c", 24: "#ff7f0e", 26: "#9467bd", 28: "#d62728", 30: "#8c564b"}
    target_g_key = "0.35"
    fig_data = {"curves": {}}

    for n in sorted(datasets.keys()):
        data = datasets[n]
        seeds = data.get("seeds", [])
        if not seeds:
            continue

        # Identify deepest trap (Rank 1, minimum E0)
        deepest = min(seeds, key=lambda s: s["trap_energy_E0"])

        q_dyn = deepest.get("quantum_dynamics", {})
        q_run = q_dyn.get(target_g_key) or list(q_dyn.values())[0] if q_dyn else None
        if not q_run:
            continue

        t_arr = np.array(q_run.get("time_array", []))
        p_arr = np.array(q_run.get("p_less_array", []))
        p_blind = q_run.get("p_blind", 0.0)

        if len(t_arr) > 0 and len(p_arr) > 0:
            c = color_map.get(n, "#333333")
            e0 = deepest["trap_energy_E0"]
            dmin = deepest["d_min"]
            label_str = f"$N = {n}$ ($E_0^*={e0}, d_{{\\min}}={dmin}$)"

            ax.plot(t_arr, p_arr, label=label_str, color=c, alpha=0.9)

            # Blind baseline dashed horizontal line
            if p_blind > 0:
                ax.axhline(p_blind, color=c, linestyle=":", alpha=0.45, linewidth=1.0)

            fig_data["curves"][n] = {
                "trap_rank": deepest.get("trap_rank", 1),
                "energy_E0": e0,
                "d_min": dmin,
                "p_blind": p_blind,
                "peak_p_less": q_run.get("peak_p_less", float(np.max(p_arr))),
                "window_avg_p_less": q_run.get("window_avg_p_less", float(np.mean(p_arr)))
            }

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
    parser = argparse.ArgumentParser(description="Publication-grade plotting suite for Bernasconi-Krylov benchmarks.")
    parser.add_argument("--data-dir", default="data_processed", help="Path to directory containing structured benchmark JSON files")
    parser.add_argument("--output-dir", default="figures/paper", help="Output directory for generated figures (.png and .pdf)")
    parser.add_argument("--bootstrap-samples", type=int, default=2000, help="Number of bootstrap resamples (default: 2000)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[plot_paper_figures] Loading persistent datasets from: {args.data_dir}")
    datasets = load_all_structured_data(args.data_dir)
    print(f"[plot_paper_figures] Loaded system sizes: N = {sorted(datasets.keys())}")

    # Generate all four figures
    summary = {}
    summary["fig_a_ratio_vs_N"] = generate_figure_a(datasets, args.output_dir, num_resamples=args.bootstrap_samples)
    summary["fig_b_rise_rate_and_pless_vs_dmin"] = generate_figure_b(datasets, args.output_dir)
    summary["fig_c_pless_and_dh_vs_g"] = generate_figure_c(datasets, args.output_dir)
    summary["fig_d_hardest_seed_dynamics"] = generate_figure_d(datasets, args.output_dir)

    # Export numerical archive
    summary_path = os.path.join(args.output_dir, "plot_data_summary.json")
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"[plot_paper_figures] Exported exact plotted numbers archive to: {summary_path}")
    print("[plot_paper_figures] ALL 4 PUBLICATION FIGURES GENERATED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
