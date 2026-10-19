#!/usr/bin/env python3
"""
Plots probability dynamics across g values for the energy-wise best Steepest Descent minima.
Generates:
  1. A dedicated standalone image for EACH individual local minimum trap (overlaying all g values).
  2. A consolidated multi-panel overview grid of all traps.
Calculates and displays the time-averaged P_<(t) over [0, t_max], Peak P_<(t), and Blind baseline.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_known_optimum(n_spins):
    opt_map = {16: 24, 20: 26, 22: 39, 24: 36, 26: 45, 28: 50, 30: 59}
    return opt_map.get(n_spins, None)

def plot_single_trap(df, trap_id, seed_key, n_spins, g_vals, out_path):
    """Generates a high-resolution standalone figure for a single trap comparing all g values."""
    trap_df = df[df[seed_key] == trap_id]
    e_seed = int(trap_df['seed_energy'].iloc[0])
    p_blind = trap_df['P_blind'].iloc[0]
    known_opt = get_known_optimum(n_spins)
    delta_e = (e_seed - known_opt) if known_opt is not None else None

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Distinct high-contrast palette for g values
    g_colors = {
        0.30: '#1f77b4',  # Blue
        0.35: '#2ca02c',  # Green
        0.45: '#d62728',  # Red
    }
    fallback_colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e', '#8c564b']

    for idx, g in enumerate(g_vals):
        gdf = trap_df[trap_df['g'] == g].sort_values('t')
        if gdf.empty:
            continue
        t = gdf['t'].values
        p_full = gdf['P_full'].values

        mean_p = np.mean(p_full[1:]) if len(p_full) > 1 else p_full[0]
        peak_p = p_full.max()
        peak_t = t[np.argmax(p_full)]

        col = g_colors.get(round(g, 2), fallback_colors[idx % len(fallback_colors)])
        label = f"$g = {g:.2f}$ (Mean: {mean_p:.2e}, Peak: {peak_p:.2e} at $t={peak_t:.1f}$s)"
        ax.plot(t, p_full, label=label, color=col, lw=2.2, alpha=0.9)

    # Blind random baseline
    if p_blind > 0:
        ax.axhline(p_blind, color='black', ls=':', lw=1.5,
                   label=f"Blind Random Baseline ($P_{{blind}} = {p_blind:.2e}$)", alpha=0.7)

    ax.set_xlabel('Evolution Time $t$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Improved Probability $P_<(t) = \sum_{E < E_0} |c_s(t)|^2$', fontsize=12, fontweight='bold')

    opt_str = f" | Known $E^* = {known_opt}$ ($\Delta E = {delta_e}$)" if known_opt is not None else ""
    ax.set_title(f"Tunneling Probability Dynamics: Trap Rank #{trap_id} ($E_0 = {e_seed}${opt_str})\nBernasconi Model ($N = {n_spins}$)",
                 fontsize=13, fontweight='bold', pad=12)

    ax.grid(True, which='both', ls='--', lw=0.6, alpha=0.4)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95, facecolor='white', edgecolor='#ccc')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_overview_grid(df, unique_traps, seed_key, n_spins, g_vals, out_path):
    """Generates a 10-panel contact sheet showing all traps for side-by-side comparison."""
    n_traps = len(unique_traps)
    ncols = 2 if n_traps <= 4 else (5 if n_traps == 10 else 3)
    nrows = (n_traps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), dpi=250, sharex=True)
    axes = np.atleast_1d(axes).flatten()

    g_colors = {0.30: '#1f77b4', 0.35: '#2ca02c', 0.45: '#d62728'}
    fallback_colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e']

    for i, trap_id in enumerate(unique_traps):
        ax = axes[i]
        trap_df = df[df[seed_key] == trap_id]
        e_seed = int(trap_df['seed_energy'].iloc[0])
        p_blind = trap_df['P_blind'].iloc[0]

        for idx, g in enumerate(g_vals):
            gdf = trap_df[trap_df['g'] == g].sort_values('t')
            if gdf.empty:
                continue
            col = g_colors.get(round(g, 2), fallback_colors[idx % len(fallback_colors)])
            ax.plot(gdf['t'].values, gdf['P_full'].values, label=f"$g={g:.2f}$", color=col, lw=1.6)

        if p_blind > 0:
            ax.axhline(p_blind, color='gray', ls=':', lw=1.0)

        ax.set_title(f"Rank #{trap_id} ($E_0={e_seed}$)", fontsize=11, fontweight='bold')
        ax.grid(True, ls='--', alpha=0.3)
        if i == 0:
            ax.legend(loc='upper left', fontsize=8)

        if i % ncols == 0:
            ax.set_ylabel(r'$P_<(t)$', fontsize=10)
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel('Time $t$', fontsize=10)

    # Hide unused subplots
    for j in range(n_traps, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Probability Dynamics $P_<(t)$ Across All {n_traps} Traps vs. $g$ ($N = {n_spins}$)",
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=250)
    plt.close(fig)

def plot_dynamics(csv_path="data_processed/probability_dynamics_g_scan.csv",
                  out_fig_or_dir=None):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find '{csv_path}'")
        return

    df = pd.read_csv(csv_path)
    g_vals = sorted(df['g'].unique())
    n_spins = int(df['N'].iloc[0])
    has_trap_rank = 'trap_rank' in df.columns
    seed_key = 'trap_rank' if has_trap_rank else 'seed_id'
    unique_traps = sorted(df[seed_key].unique())
    t_max = df['t'].max()

    # Determine output directory for individual images
    if out_fig_or_dir is None:
        out_dir = f"figures/n{n_spins}"
    elif os.path.splitext(out_fig_or_dir)[1]:
        # User passed a file path like figures/foo.png -> put individual images in figures/foo_traps/
        base_dir = os.path.dirname(out_fig_or_dir) or "figures"
        out_dir = os.path.join(base_dir, f"n{n_spins}")
    else:
        out_dir = out_fig_or_dir
    os.makedirs(out_dir, exist_ok=True)

    has_transverse = (df['P_transverse'].max() > 1e-12)

    print("\n" + "="*115)
    print(f"  PROBABILITY DYNAMICS FOR LOCAL MINIMA: N = {n_spins} | t in [0, {t_max:.1f}] | {len(unique_traps)} Trap(s)")
    print("="*115)
    print(f"  {'Trap':>7} | {'Energy':>6} | {'g':>5} | {'P_blind':>11} | {'Mean P_full [0,60]':>20} | {'Peak P_full':>13} | {'Peak t (s)':>10}")
    print("-"*115)

    # Print summary table and generate individual images per trap
    individual_files = []
    for trap_id in unique_traps:
        trap_df = df[df[seed_key] == trap_id]
        e_seed = int(trap_df['seed_energy'].iloc[0])

        for g in g_vals:
            gdf = trap_df[trap_df['g'] == g].sort_values('t')
            if gdf.empty:
                continue
            t = gdf['t'].values
            p_full = gdf['P_full'].values
            p_blind = gdf['P_blind'].iloc[0]

            mean_p_full = np.mean(p_full[1:]) if len(p_full) > 1 else p_full[0]
            peak_p_full = p_full.max()
            peak_t = t[np.argmax(p_full)]

            print(f"  Rank #{trap_id:>2} | {e_seed:>6} | {g:5.2f} | {p_blind:11.2e} | {mean_p_full:20.4e} | {peak_p_full:13.4e} | {peak_t:10.2f}")

        # Save dedicated individual image for this trap
        trap_filename = f"prob_dynamics_rank{trap_id:02d}_E{e_seed}.png"
        trap_path = os.path.join(out_dir, trap_filename)
        plot_single_trap(df, trap_id, seed_key, n_spins, g_vals, trap_path)
        individual_files.append(trap_path)

    # Also generate the consolidated overview grid
    overview_path = os.path.join(out_dir, f"prob_dynamics_all_{len(unique_traps)}_traps_grid.png")
    plot_overview_grid(df, unique_traps, seed_key, n_spins, g_vals, overview_path)

    print("="*115)
    print(f"[SUCCESS] Generated {len(individual_files)} individual trap images in: {out_dir}/")
    for f in individual_files[:3]:
        print(f"  - {f}")
    if len(individual_files) > 3:
        print(f"  - ... ({len(individual_files) - 3} more individual trap images)")
    print(f"[SUCCESS] Generated consolidated overview grid: {overview_path}\n")

if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else "data_processed/probability_dynamics_n30.csv"
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    plot_dynamics(csv_arg, out_arg)
