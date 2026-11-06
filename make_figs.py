# -*- coding: utf-8 -*-
"""生成报告用图：训练曲线与消融热力图。"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs_distill")
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

teacher = json.load(open(os.path.join(OUT, "results_teacher.json"), encoding="utf-8"))["teacher"]
base = json.load(open(os.path.join(HERE, "_tmp_results_all_baseline.json"), encoding="utf-8"))

fig, ax = plt.subplots(figsize=(6.4, 3.4))
for name, key, style in (("Teacher", teacher, "-"), ("Student-scratch", base["scratch"], "--"),
                         ("Student-KD", base["kd"], "-.")):
    c = name and key["curve"]
    ax.plot(range(1, len(c) + 1), c, style, label=name, linewidth=1.3)
ax.set_xlabel("iteration")
ax.set_ylabel("training loss")
ax.set_title("training loss: teacher vs. scratch vs. KD")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_curves.png"), dpi=160)

with open(os.path.join(OUT, "results_ablate.json"), encoding="utf-8") as f:
    abl = json.load(f)["ablation"]
Ts = sorted({r["T"] for r in abl})
As = sorted({r["alpha"] for r in abl})
m = [[next(r["val_loss"] for r in abl if r["T"] == t and r["alpha"] == a) for a in As] for t in Ts]

fig, ax = plt.subplots(figsize=(4.4, 3.6))
im = ax.imshow(m, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(len(As)), ["α=%.1f" % a for a in As])
ax.set_yticks(range(len(Ts)), ["T=%d" % t for t in Ts])
for i in range(len(Ts)):
    for j in range(len(As)):
        ax.text(j, i, "%.4f" % m[i][j], ha="center", va="center", fontsize=9)
ax.set_title("val loss heatmap (T × α)")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_heatmap.png"), dpi=160)

# 折线版消融曲线（按 alpha 分组）
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for i, t in enumerate(Ts):
    ax.plot(As, m[i], marker="o", linewidth=1.3, label="T=%d" % t)
ax.set_xlabel("alpha")
ax.set_ylabel("val loss")
ax.set_title("val loss vs. alpha for each T")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2b_ablation_lines.png"), dpi=160)
print("figures saved")
