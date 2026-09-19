import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

df = pd.read_csv(
    "data_raw/spectrum/sample_eigen.txt",
    sep=r"\s+",
    header=None,
    names=["g", "e", "data"]
)

# Sort by g, then e; preserve duplicates and original order
df = df.sort_values(["g", "e"], kind="stable").reset_index(drop=True)

# -------------------------
# Plot
# -------------------------

x = range(len(df))

# Color based on g
norm = mpl.colors.Normalize(
    vmin=df["g"].min(),
    vmax=df["g"].max()
)
cmap = plt.cm.turbo

bar_colors = cmap(norm(df["g"].values))

plt.figure(figsize=(16, 7))

plt.bar(
    x,
    df["data"],
    color=bar_colors
)

# Remove x-axis labels
plt.xticks([])

plt.xlabel("Data points")
plt.ylabel("Data point")
plt.title("Data points by (g, e)")

# Colorbar
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

plt.colorbar(sm, ax=plt.gca(), label="g")

plt.tight_layout()

# Render plot FIRST
plt.show()


# -------------------------
# Print sums
# -------------------------

sums = df.groupby(["g", "e"], sort=True)["data"].sum()

print("\nSums of data points:\n")

for g in df["g"].unique():

    print(f"g = {g}")

    # All e values belonging to this g
    e_values = sums.loc[g].index

    print("e:   ", *e_values)
    print("sum: ", *(sums.loc[g].values))

    print()