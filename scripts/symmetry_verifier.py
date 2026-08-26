#!/usr/bin/env python3

import os
import sys
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Bit transformations
# ============================================================

def reverse_bits(x: int, n: int) -> int:
    """Reverse the lowest n bits of x."""
    rev = 0
    for _ in range(n):
        rev = (rev << 1) | (x & 1)
        x >>= 1
    return rev


def complement_bits(x: int, n: int) -> int:
    """One's complement in n bits."""
    return x ^ ((1 << n) - 1)


def flip_odd_positions(x: int, n: int) -> int:
    """Flip bits at positions 1, 3, 5, ... (0-indexed from LSB)."""
    mask = 0
    for i in range(1, n, 2):
        mask |= (1 << i)
    return x ^ mask


def gray_to_binary(g: int) -> int:
    """Convert Gray code value to its binary index."""
    b = g
    while g:
        g >>= 1
        b ^= g
    return b


# ============================================================
# Formatting helpers
# ============================================================

def bits(x: int, n: int) -> str:
    return f"{x:0{n}b}"


def section(title: str) -> None:
    width = 68
    print()
    print("=" * width)
    print(title.center(width))
    print("=" * width)


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <n>")
        sys.exit(1)

    n = int(sys.argv[1])

    csv_path = "data_raw/classical/exhaustive/" + os.path.join(f"{n}.csv")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    required = {"Sequence", "E(s)"}
    if not required.issubset(df.columns):
        print(f"CSV must contain columns: {required}")
        sys.exit(1)

    df["Sequence"] = df["Sequence"].astype(int)

    energy_map = dict(zip(df["Sequence"], df["E(s)"]))

    expected_states = 1 << n

    section(f"SYSTEM INFORMATION (N = {n})")

    print(f"States in CSV       : {len(energy_map)}")
    print(f"Expected states     : {expected_states}")

    if len(energy_map) != expected_states:
        print("WARNING: CSV does not contain all 2^N states.")

    # ------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------

    section("ENERGY DISTRIBUTION")

    plt.figure(figsize=(8, 5))
    plt.hist(df["E(s)"], bins="auto", edgecolor="black")
    plt.title(f"Energy Distribution E(s), N={n}")
    plt.xlabel("E(s)")
    plt.ylabel("Number of states")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Gray code traversal
    # ------------------------------------------------------------

    section("GRAY CODE TRAVERSAL")

    df_gray = df.copy()
    df_gray["GrayOrder"] = df_gray["Sequence"].apply(gray_to_binary)
    df_gray = df_gray.sort_values("GrayOrder")

    plt.figure(figsize=(12, 5))
    plt.plot(
        df_gray["GrayOrder"],
        df_gray["E(s)"],
        linewidth=0.5
    )
    plt.title(f"E(s) Along Gray Code Traversal, N={n}")
    plt.xlabel("Gray code traversal index")
    plt.ylabel("E(s)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Symmetry definitions
    # ------------------------------------------------------------

    transforms = {
        "One's Complement": lambda x: complement_bits(x, n),
        "Bit Reversal": lambda x: reverse_bits(x, n),
        "Flip Odd Positions": lambda x: flip_odd_positions(x, n),
    }

    # ------------------------------------------------------------
    # Symmetry verification
    # ------------------------------------------------------------

    section("SYMMETRY VERIFICATION")

    all_passed = True

    for name, transform in transforms.items():

        mismatches = []
        checked = set()

        for seq, energy in energy_map.items():

            transformed = transform(seq)

            if transformed not in energy_map:
                mismatches.append(
                    (
                        seq,
                        transformed,
                        energy,
                        "MISSING",
                    )
                )
                continue

            pair = tuple(sorted((seq, transformed)))

            if pair in checked:
                continue

            checked.add(pair)

            transformed_energy = energy_map[transformed]

            if energy != transformed_energy:
                mismatches.append(
                    (
                        seq,
                        transformed,
                        energy,
                        transformed_energy,
                    )
                )

        status = "PASS" if not mismatches else "FAIL"

        print(f"{name:<25} {status}")

        if mismatches:
            all_passed = False

            for seq, transformed, e1, e2 in mismatches[:10]:
                print(
                    f"    {bits(seq, n)} -> {bits(transformed, n)}"
                    f" | E = {e1}, E' = {e2}"
                )

            if len(mismatches) > 10:
                print(
                    f"    ... {len(mismatches) - 10} additional mismatches"
                )

    print()

    if all_passed:
        print("All tested transformations preserve E(s).")
    else:
        print("WARNING: At least one transformation is not a valid symmetry.")

    # ------------------------------------------------------------
    # Union-Find construction of symmetry orbits
    # ------------------------------------------------------------

    section("SYMMETRY ORBITS")

    parent = {s: s for s in energy_map}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)

        if ra != rb:
            parent[rb] = ra

    # Union each state with all of its symmetry transforms.
    # This automatically includes closure under compositions.
    for s in energy_map:
        for transform in transforms.values():
            t = transform(s)

            if t in parent:
                union(s, t)

    # Collect complete orbits.
    orbits = defaultdict(list)

    for s in parent:
        orbits[find(s)].append(s)

    # ------------------------------------------------------------
    # Canonical representatives
    # ------------------------------------------------------------

    representatives = {}
    canonical_orbits = []

    for orbit in orbits.values():

        orbit.sort()

        rep = orbit[0]       # Minimum integer = canonical representative.

        for s in orbit:
            representatives[s] = rep

        canonical_orbits.append((rep, orbit))

    canonical_orbits.sort()

    print(f"Total states              : {len(parent)}")
    print(f"Number of symmetry orbits : {len(canonical_orbits)}")
    print(
        f"Compression ratio         : "
        f"{len(parent) / len(canonical_orbits):.3f}"
    )

    # ------------------------------------------------------------
    # Orbit size distribution
    # ------------------------------------------------------------

    section("ORBIT SIZE DISTRIBUTION")

    sizes = defaultdict(int)

    for _, orbit in canonical_orbits:
        sizes[len(orbit)] += 1

    print(f"{'Orbit size':>12} | {'Number of orbits':>18}")
    print("-" * 35)

    for size in sorted(sizes):
        print(f"{size:>12} | {sizes[size]:>18}")

    # ------------------------------------------------------------
    # Canonical representatives
    # ------------------------------------------------------------

    section("CANONICAL SYMMETRY REPRESENTATIVES")

    print(
        f"{'Representative':>{n}} | "
        f"{'Integer':>10} | "
        f"{'Orbit size':>10} | "
        f"{'Energy':>12}"
    )

    print("-" * (n + 42))

    for rep, orbit in canonical_orbits:

        print(
            f"{bits(rep, n):>{n}} | "
            f"{rep:>10} | "
            f"{len(orbit):>10} | "
            f"{energy_map[rep]:>12}"
        )

    # ------------------------------------------------------------
    # Optional complete state -> representative mapping
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Representatives and number of states represented
    # ------------------------------------------------------------
    
    section("SYMMETRY REPRESENTATIVE SUMMARY")
    
    print(
        f"{'Representative':>{n}} | "
        f"{'Integer':>10} | "
        f"{'States represented':>18} | "
        f"{'Energy':>12}"
    )
    
    print("-" * (n + 50))
    
    for rep, orbit in canonical_orbits:
        print(
            f"{bits(rep, n):>{n}} | "
            f"{rep:>10} | "
            f"{len(orbit):>18} | "
            f"{energy_map[rep]:>12}"
        )
    

if __name__ == "__main__":
    main()