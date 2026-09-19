import matplotlib.pyplot as plt
from collections import defaultdict

data = defaultdict(list)

with open("data_raw/spectrum/samplerun.txt", "r") as f:
    for line in f:
        x, y = map(float, line.split())
        data[x].append(y)

xs = sorted(data.keys())
rng = int(input("Enter number of eigenvalues to plot: "))
for i in range(rng):
    ys = [data[x][i] for x in xs]
    plt.plot(xs, ys, marker="o", label=f"y{i+1}")

plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()