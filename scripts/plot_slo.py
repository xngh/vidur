import matplotlib.pyplot as plt

qps = [5, 6, 7, 8, 9]

map_reduce_data = {
    "Round Robin": [0.7850, 0.6970, 0.6141, 0.4512, 0.3865],
    "Least Used":  [0.7026, 0.6032, 0.5553, 0.5075, 0.4619],
    "Parrot":      [0.7915, 0.7230, 0.6254, 0.5934, 0.5712],
    "SOLWE":       [0.9108, 0.8213, 0.7145, 0.6764, 0.6090],
}

colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
markers = ["o", "s", "^", "D"]

fig, ax = plt.subplots(figsize=(7, 5))

for (label, values), color, marker in zip(map_reduce_data.items(), colors, markers):
    ax.plot(qps, values, label=label, color=color, marker=marker,
            linewidth=2, markersize=7)

ax.set_xlabel("到达率（工作流/s）", fontsize=13)
ax.set_ylabel("SLO满足率", fontsize=13)
# ax.set_title("SLO Satisfaction Rate vs QPS", fontsize=14)
ax.set_xticks(qps)
ax.set_ylim(0.3, 1.0)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("slo_satisfaction_rate.pdf")
