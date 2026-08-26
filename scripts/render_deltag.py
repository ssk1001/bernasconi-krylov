#!/usr/bin/env python3

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt


def get_n_from_filename(filename):
    """Extract N from filenames like gap_N4.csv."""
    name = os.path.basename(filename)

    try:
        start = name.index("gap_N") + len("gap_N")
        end = name.index(".csv", start)
        return int(name[start:end])
    except (ValueError, IndexError):
        return name


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} gap_N4.csv gap_N5.csv ...")
        sys.exit(1)

    plt.figure(figsize=(10, 6))

    for filename in sys.argv[1:]:
        filename = "data_raw/spectral_gap/" + filename
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found, skipping.")
            continue

        df = pd.read_csv(filename)

        required = {"g", "E0", "E1", "Delta"}
        if not required.issubset(df.columns):
            print(
                f"Warning: {filename} does not contain "
                f"the required columns {required}, skipping."
            )
            continue

        n = get_n_from_filename(filename)
        df = df.sort_values("g").reset_index(drop=True)

        # Plot the spectral gap.
        plt.plot(
            df["g"],
            df["Delta"],
            marker="o",
            markersize=3,
            label=f"N = {n}"
        )

        # Find the minimum gap.
        min_idx = df["Delta"].idxmin()
        min_g = df.loc[min_idx, "g"]
        min_delta = df.loc[min_idx, "Delta"]

        # Mark the minimum.
        plt.scatter(
            min_g,
            min_delta,
            s=70,
            zorder=5
        )

        # Annotate it.
        plt.annotate(
            f"N={n}\n"
            f"g={min_g:.2f}\n"
            f"Δ={min_delta:.6g}",
            xy=(min_g, min_delta),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=8
        )

        # Also print it to the terminal.
        print(
            f"N = {n}: "
            f"minimum gap = {min_delta:.16g} "
            f"at g = {min_g:.16g}"
        )

    plt.xlabel("g")
    plt.ylabel(r"$\Delta(g) = E_1(g) - E_0(g)$")
    plt.title("Lowest Spectral Gap vs Coupling Strength")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()