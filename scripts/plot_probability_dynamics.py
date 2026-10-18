#!/usr/bin/env python3
"""
Plots probability dynamics across g values for the energy-wise best Steepest Descent minima.
Calculates and displays the time-averaged P_<(t) over [0, t_max], Peak P_<(t), and Blind baseline.
Supports both dual-panel (with Transverse baseline) and single-panel (Full H alone) modes.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_dynamics(csv_path="data_processed/probability_dynamics_g_scan.csv",
                  out_fig=None):
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

    if out_fig is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        out_fig = f"figures/{base_name}.png"
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)

    has_transverse = (df['P_transverse'].max() > 1e-12)

    print("\n" + "="*115)
    print(f"  PROBABILITY DYNAMICS FOR LOCAL MINIMA: N = {n_spins} | t in [0, {t_max:.1f}] | {len(unique_traps)} Trap(s)")
    print("="*115)
    if has_transverse:
        print(f"  {'g':>5} | {'Trap':>7} | {'Energy':>6} | {'P_blind':>11} | {'Mean P_full':>13} | {'Peak P_full':>13} | {'Peak t (s)':>10} | {'Peak P_tr':>11} | {'Max ΔP':>8}")
    else:
        print(f"  {'g':>5} | {'Trap':>7} | {'Energy':>6} | {'P_blind':>11} | {'Mean P_full':>13} | {'Peak P_full':>13} | {'Peak t (s)':>10}")
    print("-"*115)

    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#ff7f0e']

    if has_transverse:
        fig, axes = plt.subplots(len(g_vals), 2, figsize=(14, 3.4 * len(g_vals)), sharex='col')
        if len(g_vals) == 1:
            axes = np.expand_dims(axes, 0)
    else:
        fig, axes = plt.subplots(len(g_vals), 1, figsize=(10, 3.2 * len(g_vals)), sharex=True)
        if len(g_vals) == 1:
            axes = np.array([axes])

    for i, g in enumerate(g_vals):
        gdf = df[df['g'] == g]
        ax1 = axes[i, 0] if has_transverse else axes[i]
        ax2 = axes[i, 1] if has_transverse else None

        for s_idx, trap_id in enumerate(unique_traps):
            sdf = gdf[gdf[seed_key] == trap_id].sort_values('t')
            t = sdf['t'].values
            p_full = sdf['P_full'].values
            p_tr = sdf['P_transverse'].values
            p_blind = sdf['P_blind'].iloc[0]
            e_seed = int(sdf['seed_energy'].iloc[0])

            # Discrete time-average over [0, t_max]
            mean_p_full = np.mean(p_full[1:]) if len(p_full) > 1 else p_full[0]
            peak_p_full = p_full.max()
            peak_t = t[np.argmax(p_full)]

            col = colors[s_idx % len(colors)]
            tag = f"Rank #{trap_id} (E={e_seed})" if len(unique_traps) > 1 else f"E0={e_seed}"

            if has_transverse:
                delta_p = p_full - p_tr
                mean_p_tr = np.mean(p_tr[1:]) if len(p_tr) > 1 else p_tr[0]
                peak_p_tr = p_tr.max()
                print(f"  {g:5.2f} | {trap_id:>7} | {e_seed:>6} | {p_blind:11.2e} | {mean_p_full:13.4e} | {peak_p_full:13.4e} | {peak_t:10.2f} | {peak_p_tr:11.4e} | {delta_p.max():8.4f}")

                # Left column: P(t) comparison
                ax1.plot(t, p_full, label=f'Full $H$ {tag} (Mean: {mean_p_full:.2e}, Peak: {peak_p_full:.2e})', color=col, lw=2)
                ax1.plot(t, p_tr, label=f'Transverse {tag}', color=col, lw=1.5, ls='--')
                ax1.axhline(p_blind, color='gray', ls=':', lw=1.2, label=f'Blind {tag} ({p_blind:.2e})' if s_idx == 0 else "")

                # Right column: Differential Steering ΔP(t)
                ax2.plot(t, delta_p, label=f'$\Delta P$ {tag}', color=col, lw=2)
            else:
                print(f"  {g:5.2f} | {trap_id:>7} | {e_seed:>6} | {p_blind:11.2e} | {mean_p_full:13.4e} | {peak_p_full:13.4e} | {peak_t:10.2f}")
                ax1.plot(t, p_full, label=f'Full $H$ {tag} (Mean: {mean_p_full:.2e}, Peak: {peak_p_full:.2e})', color=col, lw=2)
                ax1.axhline(p_blind, color='gray', ls=':', lw=1.2, label=f'Blind {tag} ({p_blind:.2e})' if s_idx == 0 else "")

        ax1.set_ylabel(r'$P(E < E_0)$', fontsize=11)
        ax1.set_title(f'$g = {g}$ (Coupling Strength)', fontsize=12, fontweight='bold', loc='left')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=9)

        if has_transverse and ax2 is not None:
            ax2.axhline(0, color='black', lw=0.8, ls='--')
            ax2.set_ylabel(r'$\Delta P = P_{\mathrm{full}} - P_{\mathrm{transv}}$', fontsize=11)
            ax2.set_title(f'Net Steering Effect ($g = {g}$)', fontsize=12, loc='left')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right', fontsize=9)

    if has_transverse:
        axes[-1, 0].set_xlabel('Time $t$', fontsize=11)
        axes[-1, 1].set_xlabel('Time $t$', fontsize=11)
    else:
        axes[-1].set_xlabel('Time $t$', fontsize=11)

    title_desc = f"Local Minimum Trap (Rank #1)" if len(unique_traps) == 1 else f"Top {len(unique_traps)} Local Minima Traps"
    plt.suptitle(f'Tunneling Probability Dynamics $P_<(t)$ for {title_desc} (Bernasconi Model, $N={n_spins}$)',
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    print("="*115)
    print(f"Figure saved to: {out_fig}\n")

if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else "data_processed/probability_dynamics_g_scan.csv"
    fig_arg = sys.argv[2] if len(sys.argv) > 2 else None
    plot_dynamics(csv_arg, fig_arg)
