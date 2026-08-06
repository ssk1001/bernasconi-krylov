#!/usr/bin/env python3

import os
import sys
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt


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
    """
    Flip bits at odd positions (0-indexed from LSB):
    positions = 1,3,5,...
    """
    mask = 0
    for i in range(1, n, 2):
        mask |= (1 << i)
    return x ^ mask

def gray_to_binary(g):
    b = g
    while g:
        g >>= 1
        b ^= g
    return b

    


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <n>")
        sys.exit(1)

    n = int(sys.argv[1])

    csv_path = os.path.join(".", "cache", f"{n}.csv")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    required = {"Sequence", "E(s)"}
    if not required.issubset(df.columns):
        print(f"CSV must contain columns: {required}")
        sys.exit(1)

    # Convert to integer if necessary
    df["Sequence"] = df["Sequence"].astype(int)

    energy_map = dict(zip(df["Sequence"], df["E(s)"]))

    # ------------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.hist(df["E(s)"], bins="auto", edgecolor="black")
    plt.title(f"E(s) Histogram (n={n})")
    plt.xlabel("E(s)")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    df_gray = df.copy()
    df_gray["GrayOrder"] = df_gray["Sequence"].apply(gray_to_binary)
    df_gray = df_gray.sort_values("GrayOrder")
    
    plt.figure(figsize=(12, 5))
    plt.plot(df_gray["GrayOrder"], df_gray["E(s)"], linewidth=0.5)
    plt.title(f"E(s) in Gray Code Traversal (n={n})")
    plt.xlabel("Gray code traversal index")
    plt.ylabel("E(s)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


    # ------------------------------------------------------------------
    # Symmetry checks
    # ------------------------------------------------------------------

    transforms = {
        "Ones Complement": lambda x: complement_bits(x, n),
        "Bit Reversal": lambda x: reverse_bits(x, n),
        "Flip Odd Positions": lambda x: flip_odd_positions(x, n),
    }

    print("\n================ Symmetry Verification ================\n")

    for name, transform in transforms.items():
        mismatches = []

        checked = set()

        for seq, energy in energy_map.items():
            transformed = transform(seq)

            if transformed not in energy_map:
                mismatches.append(
                    (seq, transformed, energy, "Missing transformed sequence")
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

        print(f"{name}")
        print("-" * len(name))

        if not mismatches:
            print("PASS\n")
        else:
            print(f"FAILED ({len(mismatches)} mismatches)")
            for m in mismatches[:20]:
                print(m)
            if len(mismatches) > 20:
                print(f"... {len(mismatches)-20} more mismatches")
            print()

    # ------------------------------------------------------------------
    # Orbit statistics (optional)
    # ------------------------------------------------------------------

    print("=============== Orbit Statistics ===============")

    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for s in energy_map:
        parent[s] = s

    for s in energy_map:
        for transform in transforms.values():
            t = transform(s)
            if t in parent:
                union(s, t)

    orbits = defaultdict(list)
    for s in parent:
        orbits[find(s)].append(s)

    print(f"Number of symmetry orbits: {len(orbits)}")

    sizes = defaultdict(int)
    for orbit in orbits.values():
        sizes[len(orbit)] += 1

    print("Orbit size distribution:")
    for size in sorted(sizes):
        print(f"  size {size}: {sizes[size]}")


if __name__ == "__main__":
    main()