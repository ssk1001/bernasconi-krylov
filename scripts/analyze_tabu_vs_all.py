#!/usr/bin/env python3
"""
Comprehensive Comparative Analysis: Tabu Search vs Classical Steepest Descent vs Quantum Walk
Evaluates per-evaluation escape rates, success probabilities, scaling, and advantage ratios
across system sizes N = 20, 22, 24, 26, 28, 30 for all available benchmark seeds.
"""

import os
import glob
import csv
import json
import numpy as np

def load_tabu_benchmarks(data_dir="data_processed"):
    tabu_data = {}
    files = sorted(glob.glob(os.path.join(data_dir, "tabu_benchmark_n*.csv")))
    for f in files:
        if "rank2" in f:
            continue
        with open(f, "r") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                n = int(row["n"])
                rk = int(row["trap_rank"])
                e0 = int(row["E0"])
                d_min = int(row["d_min"])
                method = row["method"].strip()
                kick_d = int(row["kick_d"])
                hits = int(row["hits"])
                trials = int(row["trials"])
                succ_rate = float(row["success_rate"])
                total_evals = int(row["total_evals"])
                per_eval_rate = float(row["per_eval_rate"])
                mean_evals = float(row["mean_evals_trial"])
                time_sec = float(row["time_sec"])

                key = (n, rk)
                if key not in tabu_data:
                    tabu_data[key] = {
                        "n": n, "rank": rk, "E0": e0, "d_min": d_min,
                        "tabu_direct": {},
                        "tabu_kicks": {},
                        "tabu_restart": {},
                        "sd_kicks": {},
                        "sd_restart": {}
                    }

                entry = {
                    "hits": hits, "trials": trials, "succ_rate": succ_rate,
                    "total_evals": total_evals, "rate": per_eval_rate,
                    "mean_evals": mean_evals, "time_sec": time_sec
                }

                if method == "Tabu Direct Escape":
                    tabu_data[key]["tabu_direct"] = entry
                elif "Tabu Kick" in method:
                    tabu_data[key]["tabu_kicks"][kick_d] = entry
                elif method == "Tabu Random Restart":
                    tabu_data[key]["tabu_restart"] = entry
                elif "SD Kick" in method:
                    tabu_data[key]["sd_kicks"][kick_d] = entry
                elif method == "SD Random Restart":
                    tabu_data[key]["sd_restart"] = entry

    # Compute summary rates for each trap
    for key, d in tabu_data.items():
        # Best tabu kick
        best_k_rate = 0.0
        best_k_d = 0
        best_k_entry = None
        for kd, entry in d["tabu_kicks"].items():
            if entry["rate"] > best_k_rate:
                best_k_rate = entry["rate"]
                best_k_d = kd
                best_k_entry = entry
        d["best_tabu_kick"] = {"d": best_k_d, "rate": best_k_rate, "entry": best_k_entry}

        # Best SD kick
        best_sd_k_rate = 0.0
        best_sd_k_d = 0
        best_sd_k_entry = None
        for kd, entry in d["sd_kicks"].items():
            if entry["rate"] > best_sd_k_rate:
                best_sd_k_rate = entry["rate"]
                best_sd_k_d = kd
                best_sd_k_entry = entry
        d["best_sd_kick"] = {"d": best_sd_k_d, "rate": best_sd_k_rate, "entry": best_sd_k_entry}

        # Tabu best overall
        r_dir = d["tabu_direct"].get("rate", 0.0)
        r_rest = d["tabu_restart"].get("rate", 0.0)
        d["tabu_best_rate"] = max(r_dir, best_k_rate, r_rest)
        if d["tabu_best_rate"] == r_dir:
            d["tabu_best_method"] = "Direct"
        elif d["tabu_best_rate"] == best_k_rate:
            d["tabu_best_method"] = f"Kick(d={best_k_d})"
        else:
            d["tabu_best_method"] = "Restart"

        # SD best overall
        sd_rest_rate = d["sd_restart"].get("rate", 0.0)
        d["sd_best_rate"] = max(best_sd_k_rate, sd_rest_rate)
        if d["sd_best_rate"] == best_sd_k_rate and best_sd_k_rate > 0:
            d["sd_best_method"] = f"Kick(d={best_sd_k_d})"
        else:
            d["sd_best_method"] = "Restart"

    return tabu_data

def load_quantum_and_classical_benchmarks(data_dir="data_processed"):
    bench_data = {}
    files = sorted(glob.glob(os.path.join(data_dir, "benchmark_results_n*.csv")))
    for f in files:
        with open(f, "r") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                n = int(row["N"])
                rk = int(row["trap_rank"])
                g = float(row["g"])
                key = (n, rk)
                if key not in bench_data:
                    bench_data[key] = {
                        "n": n, "rank": rk,
                        "E0": int(row["trap_energy"]),
                        "d_min": int(row["d_min"]),
                        "trap_bits": row.get("trap_bits", ""),
                        "best_d": int(row["best_d"]),
                        "best_d_rate": float(row["best_d_rate"]),
                        "restart_rate": float(row["restart_rate"]),
                        "classical_best_rate": max(float(row["best_d_rate"]), float(row["restart_rate"])),
                        "quantum_runs": {}
                    }
                bench_data[key]["quantum_runs"][g] = {
                    "g": g,
                    "window_avg_p_less": float(row["window_avg_p_less"]),
                    "initial_rise_rate_avg": float(row["initial_rise_rate_avg"]),
                    "initial_rise_rate_max": float(row["initial_rise_rate_max"]),
                    "peak_p_less": float(row["peak_p_less"]),
                    "p_blind": float(row["p_blind"]),
                    "window_avg_dh": float(row.get("window_avg_dh", 0.0))
                }
    return bench_data

def analyze():
    tabu_data = load_tabu_benchmarks()
    bench_data = load_quantum_and_classical_benchmarks()

    print("========================================================================================================================")
    print("                    COMPREHENSIVE BENCHMARK: TABU SEARCH vs CLASSICAL STEEPEST DESCENT vs QUANTUM WALK                 ")
    print("========================================================================================================================")

    known_opts = {16: 24, 20: 26, 22: 39, 24: 36, 26: 45, 28: 50, 30: 59}

    all_keys = sorted(tabu_data.keys(), key=lambda x: (x[0], x[1]))
    
    # Store aggregated rows
    rows_summary = []

    for (n, rk) in all_keys:
        td = tabu_data[(n, rk)]
        bd = bench_data.get((n, rk), {})

        e0 = td["E0"]
        dmin = td["d_min"]
        e_opt = known_opts.get(n, 0)
        delta_e = e0 - e_opt

        # Tabu metrics
        t_dir_rate = td["tabu_direct"].get("rate", 0.0)
        t_dir_succ = td["tabu_direct"].get("succ_rate", 0.0)
        t_kick_rate = td["best_tabu_kick"]["rate"]
        t_kick_d = td["best_tabu_kick"]["d"]
        t_rest_rate = td["tabu_restart"].get("rate", 0.0)
        t_best_rate = td["tabu_best_rate"]
        t_best_method = td["tabu_best_method"]

        # Classical SD metrics (from local tabu run and validated against benchmark CSV)
        sd_kick_rate = td["best_sd_kick"]["rate"]
        sd_kick_d = td["best_sd_kick"]["d"]
        sd_rest_rate = td["sd_restart"].get("rate", 0.0)
        sd_best_rate = td["sd_best_rate"]

        # Benchmark CSV values if available
        if bd:
            csv_best_d = bd["best_d"]
            csv_best_d_rate = bd["best_d_rate"]
            csv_rest_rate = bd["restart_rate"]
            csv_c_best = bd["classical_best_rate"]
        else:
            csv_best_d = sd_kick_d
            csv_best_d_rate = sd_kick_rate
            csv_rest_rate = sd_rest_rate
            csv_c_best = sd_best_rate

        # Quantum metrics (evaluated at standard g=0.35)
        q_g035 = bd.get("quantum_runs", {}).get(0.35, {}) if bd else {}
        if not q_g035 and bd.get("quantum_runs"):
            # fallback to first available g
            q_g035 = list(bd["quantum_runs"].values())[0]

        q_win_p = q_g035.get("window_avg_p_less", 0.0)
        q_rise_rate = q_g035.get("initial_rise_rate_avg", 0.0)
        q_peak_p = q_g035.get("peak_p_less", 0.0)
        q_p_blind = q_g035.get("p_blind", 0.0)

        # Comparative Ratios
        tabu_vs_sd_ratio = (t_best_rate / csv_c_best) if csv_c_best > 0 else (999.0 if t_best_rate > 0 else 1.0)
        tabu_direct_vs_sd_ratio = (t_dir_rate / csv_c_best) if csv_c_best > 0 else (999.0 if t_dir_rate > 0 else 1.0)
        
        q_vs_sd_ratio = (q_win_p / csv_c_best) if (csv_c_best > 0 and q_win_p > 0) else (999.0 if q_win_p > 0 else 0.0)
        q_vs_tabu_ratio = (q_win_p / t_best_rate) if (t_best_rate > 0 and q_win_p > 0) else 0.0
        q_vs_tabu_direct = (q_win_p / t_dir_rate) if (t_dir_rate > 0 and q_win_p > 0) else 0.0

        rows_summary.append({
            "n": n, "rank": rk, "E0": e0, "d_min": dmin, "delta_E": delta_e,
            "tabu_direct_rate": t_dir_rate, "tabu_direct_succ": t_dir_succ,
            "tabu_kick_rate": t_kick_rate, "tabu_kick_d": t_kick_d,
            "tabu_restart_rate": t_rest_rate,
            "tabu_best_rate": t_best_rate, "tabu_best_method": t_best_method,
            "sd_best_rate": csv_c_best, "sd_best_d": csv_best_d,
            "sd_restart_rate": csv_rest_rate,
            "q_win_p": q_win_p, "q_rise_rate": q_rise_rate, "q_peak_p": q_peak_p, "q_p_blind": q_p_blind,
            "tabu_vs_sd": tabu_vs_sd_ratio,
            "q_vs_sd": q_vs_sd_ratio,
            "q_vs_tabu": q_vs_tabu_ratio,
            "q_vs_tabu_direct": q_vs_tabu_direct
        })

    # Print summary per N
    for n in sorted(list(set(r["n"] for r in rows_summary))):
        n_rows = [r for r in rows_summary if r["n"] == n]
        print(f"\n========================================================================================================================")
        print(f"                               SYSTEM SIZE N = {n} (Top {len(n_rows)} Benchmark Traps)")
        print(f"========================================================================================================================")
        print(f" Rk | E0 | d_m | Tabu Direct | Tabu Best (meth)  | SD Best (meth)   | Q Win P_<   | Q Rise dP/dt| Tabu/SD | Q/SD  | Q/Tabu")
        print(f"----+----+-----+-------------+-------------------+------------------+-------------+-------------+---------+-------+-------")
        for r in n_rows:
            q_p_str = f"{r['q_win_p']:.2e}" if r['q_win_p'] > 0 else "   N/A    "
            q_r_str = f"{r['q_rise_rate']:.2e}" if r['q_rise_rate'] > 0 else "   N/A    "
            q_sd_str = f"{r['q_vs_sd']:.2f}x" if r['q_win_p'] > 0 else "  N/A "
            q_tabu_str = f"{r['q_vs_tabu']:.2f}x" if r['q_win_p'] > 0 else "  N/A "
            
            t_meth_str = f"{r['tabu_best_rate']:.2e} ({r['tabu_best_method'][:6]})"
            sd_meth_str = f"{r['sd_best_rate']:.2e} (d={r['sd_best_d']})" if r['sd_best_rate'] > 0 else "0.00e+00 (none)"

            print(f" {r['rank']:2d} | {r['E0']:2d} | {r['d_min']:3d} |  {r['tabu_direct_rate']:.2e}   | {t_meth_str:17s} | {sd_meth_str:16s} |  {q_p_str:10s} |  {q_r_str:10s} | {r['tabu_vs_sd']:6.2f}x |{q_sd_str:6s} |{q_tabu_str:6s}")

    # Compute overall statistics
    print("\n========================================================================================================================")
    print("                                      AGGREGATED METRICS & CROSS-METHOD COMPARISON                                     ")
    print("========================================================================================================================")
    print(" N  | Traps | Med Tabu Direct | Med Tabu Best | Med SD Best   | Med Q Win P_< | Med Tabu/SD | Med Q/SD   | Med Q/Tabu")
    print("----+-------+-----------------+---------------+---------------+---------------+-------------+------------+------------")
    for n in sorted(list(set(r["n"] for r in rows_summary))):
        n_rows = [r for r in rows_summary if r["n"] == n]
        med_t_dir = np.median([r["tabu_direct_rate"] for r in n_rows])
        med_t_best = np.median([r["tabu_best_rate"] for r in n_rows])
        med_sd_best = np.median([r["sd_best_rate"] for r in n_rows])
        
        q_vals = [r["q_win_p"] for r in n_rows if r["q_win_p"] > 0]
        med_q = np.median(q_vals) if q_vals else 0.0
        
        t_sd_ratios = [r["tabu_vs_sd"] for r in n_rows if r["tabu_vs_sd"] < 900]
        med_t_sd = np.median(t_sd_ratios) if t_sd_ratios else 0.0
        
        q_sd_ratios = [r["q_vs_sd"] for r in n_rows if r["q_vs_sd"] > 0 and r["q_vs_sd"] < 900]
        med_q_sd = np.median(q_sd_ratios) if q_sd_ratios else 0.0
        
        q_t_ratios = [r["q_vs_tabu"] for r in n_rows if r["q_vs_tabu"] > 0]
        med_q_t = np.median(q_t_ratios) if q_t_ratios else 0.0

        q_str = f"{med_q:.3e}" if med_q > 0 else "   N/A    "
        q_sd_s = f"{med_q_sd:6.2f}x" if med_q_sd > 0 else "  N/A  "
        q_t_s = f"{med_q_t:6.2f}x" if med_q_t > 0 else "  N/A  "

        print(f" {n:2d} |  {len(n_rows):2d}   |    {med_t_dir:.3e}    |   {med_t_best:.3e}   |   {med_sd_best:.3e}   |  {q_str:12s} |   {med_t_sd:5.2f}x    |  {q_sd_s:8s}  |  {q_t_s:8s}")

    # Deepest Traps (Rank 1) scaling analysis
    print("\n========================================================================================================================")
    print("                         HARDEST METASTABLE STATES (RANK 1 / DEEPEST TRAPS) SCALING ACROSS N                            ")
    print("========================================================================================================================")
    print(" N  | E0 | d_m | Tabu Direct | Tabu Best     | SD Best (d)   | Q Win P_<     | Q Rise dP/dt  | Tabu/SD | Q/SD   | Q/Tabu")
    print("----+----+-----+-------------+---------------+---------------+---------------+---------------+---------+--------+-------")
    rank1_rows = [r for r in rows_summary if r["rank"] == 1]
    for r in rank1_rows:
        q_p_str = f"{r['q_win_p']:.3e}" if r['q_win_p'] > 0 else "   N/A    "
        q_r_str = f"{r['q_rise_rate']:.3e}" if r['q_rise_rate'] > 0 else "   N/A    "
        q_sd_str = f"{r['q_vs_sd']:6.2f}x" if r['q_win_p'] > 0 else "  N/A  "
        q_tabu_str = f"{r['q_vs_tabu']:6.2f}x" if r['q_win_p'] > 0 else "  N/A  "
        sd_str = f"{r['sd_best_rate']:.3e} (d={r['sd_best_d']})" if r['sd_best_rate'] > 0 else "0.000e+00 (none)"
        print(f" {r['n']:2d} | {r['E0']:2d} |  {r['d_min']}  |  {r['tabu_direct_rate']:.3e}  |   {r['tabu_best_rate']:.3e}   | {sd_str:13s} |   {q_p_str:11s} |   {q_r_str:11s} | {r['tabu_vs_sd']:6.2f}x | {q_sd_str:6s} | {q_tabu_str:6s}")

    # Save complete merged data to JSON and CSV
    out_json = "data_processed/tabu_vs_classical_quantum_comparison.json"
    out_csv = "data_processed/tabu_vs_classical_quantum_comparison.csv"
    
    with open(out_json, "w") as fp:
        json.dump(rows_summary, fp, indent=2)
    print(f"\n[Saved] Comprehensive JSON exported to: {out_json}")

    with open(out_csv, "w", newline="") as fp:
        fieldnames = list(rows_summary[0].keys())
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_summary)
    print(f"[Saved] Comprehensive CSV exported to: {out_csv}")

if __name__ == "__main__":
    analyze()
