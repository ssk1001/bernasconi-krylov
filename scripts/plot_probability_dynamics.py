#!/usr/bin/env python3
"""
Plots probability dynamics across g values for the energy-wise best Steepest Descent minima.
Compares Full Quantum Evolution, Transverse Field Alone, and Blind Random Sampling per minimum.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_dynamics(csv_path="data_processed/probability_dynamics_g_scan.csv",
                  out_fig="figures/probability_dynamics_g_scan.png"):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find '{csv_path}'")
        return

    df = pd.read_csv(csv_path)
    g_vals = sorted(df['g'].unique())
    n_spins = int(df['N'].iloc[0])
    seed_ids = sorted(df['seed_id'].unique())
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)

    fig, axes = plt.subplots(len(g_vals), 2, figsize=(14, 3.2 * len(g_vals)), sharex='col')
    if len(g_vals) == 1:
        axes = np.expand_dims(axes, 0)

    print("\n" + "="*90)
    print(f"  PROBABILITY DYNAMICS FOR BEST LOCAL MINIMA (N = {n_spins}, {len(seed_ids)} Evaluated Minima)")
    print("="*90)
    print(f"  {'g':>5} | {'Min ID':>6} | {'Energy':>6} | {'P_blind':>12} | {'Peak P_full':>12} | {'Peak P_tr':>12} | {'Max ΔP':>8}")
    print("-"*90)

    colors = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b', '#e377c2']

    for i, g in enumerate(g_vals):
        gdf = df[df['g'] == g]
        ax1 = axes[i, 0]
        ax2 = axes[i, 1]

        for s_idx, s_id in enumerate(seed_ids):
            sdf = gdf[gdf['seed_id'] == s_id].sort_values('t')
            t = sdf['t'].values
            p_full = sdf['P_full'].values
            p_tr = sdf['P_transverse'].values
            p_blind = sdf['P_blind'].iloc[0]
            e_seed = int(sdf['seed_energy'].iloc[0])
            delta_p = p_full - p_tr

            col = colors[s_idx % len(colors)]
            tag = f"(E={e_seed})" if len(seed_ids) == 1 else f"(Min #{s_id+1}, E={e_seed})"

            print(f"  {g:5.2f} | {s_id+1:>6} | {e_seed:>6} | {p_blind:12.3e} | {p_full.max():12.4f} | {p_tr.max():12.4f} | {delta_p.max():8.4f}")

            # Left column: P(t) comparison
            ax1.plot(t, p_full, label=f'Full $H$ {tag}', color=col, lw=2)
            ax1.plot(t, p_tr, label=f'Transverse {tag}', color=col, lw=1.8, ls='--')
            ax1.axhline(p_blind, color='gray', ls=':', lw=1.2, label=f'Blind {tag} ({p_blind:.2e})' if s_idx == 0 else "")

            # Right column: Differential Steering ΔP(t)
            ax2.plot(t, delta_p, label=f'$\Delta P$ {tag}', color=col, lw=2)

        ax1.set_ylabel(r'$P(E < E_0)$', fontsize=11)
        ax1.set_title(f'$g = {g}$ (Coupling Strength)', fontsize=12, fontweight='bold', loc='left')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=9)

        ax2.axhline(0, color='black', lw=0.8, ls='--')
        ax2.set_ylabel(r'$\Delta P = P_{\mathrm{full}} - P_{\mathrm{transv}}$', fontsize=11)
        ax2.set_title(f'Net Steering Effect ($g = {g}$)', fontsize=12, loc='left')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=9)

    axes[-1, 0].set_xlabel('Time $t$', fontsize=11)
    axes[-1, 1].set_xlabel('Time $t$', fontsize=11)

    title_desc = f"Best Local Minimum" if len(seed_ids) == 1 else f"Top {len(seed_ids)} Local Minima"
    plt.suptitle(f'Tunneling Probability Dynamics for {title_desc} vs. $g$ (Bernasconi Model, $N={n_spins}$)',
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    print("="*90)
    print(f"Figure saved to: {out_fig}\n")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data_processed/probability_dynamics_g_scan.csv"
    plot_dynamics(path)
