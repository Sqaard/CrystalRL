# Figures for "CrystalRL — Crystal Clear Reinforcement Learning".
# Every number is copied from a repo artifact, cited in the panel comment.
# Palette = the session-validated dataviz slots; surface #fcfcfb.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2a78d6"; LIGHTBLUE = "#a8c8ef"; GREEN = "#008300"; AMBER = "#eda100"
PURPLE = "#4a3aa7"; SURF = "#fcfcfb"; INK = "#222222"; MUT = "#666666"; RED = "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 11, "axes.edgecolor": MUT, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------- F1: CrystalScore
# Source: reports/CRYSTAL1_CSI500_AND_CRYSTALSCORE.md (the identical-scalar table).
fig, ax = plt.subplots(figsize=(8.6, 4.0))
axes_names = ["Faithfulness", "Simulatability", "Stability", "CrystalScore\n(product, K≤9)"]
r6c = [1.00, 0.244, 0.619, 0.151]
c1 = [1.00, 0.938, 1.00, 0.938]
x = np.arange(4); w = 0.36
b1 = ax.bar(x - w / 2, r6c, w, color=LIGHTBLUE, label="R6c (CHRL line, 64-d latent)")
b2 = ax.bar(x + w / 2, c1, w, color=BLUE, label="CRYSTAL-1 (M6 soft-tree head)")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
ax.set_xticks(x); ax.set_xticklabels(axes_names)
ax.set_ylim(0, 1.12); ax.set_ylabel("score (identical scalar, both policies)")
ax.set_title("Born-legible beats explained-later: the entire 6× gap is Simulatability", fontsize=12)
ax.legend(loc="center left", frameon=False, fontsize=10)
fig.tight_layout(); fig.savefig("figures/fig_crystalscore.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- F2: the two architectures
# Schematic (no numbers): CHRL's explained-later stack vs CRYSTAL-1's born-legible stack.
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
for ax in axes:
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

def box(ax, x, y, w, h, text, ec, fc="white", fs=9.5, lw=1.6):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)

ax = axes[0]
ax.set_title("CHRL / R6c — transparent LAYERS, opaque core", fontsize=12, color=INK)
box(ax, 0.4, 7.6, 9.2, 1.6, "3 readable layers: risk (20d) → groups → stocks (5d)\n+ guardrails: pacing, confidence scaling, top-K, buy gate", BLUE)
box(ax, 0.4, 4.4, 9.2, 2.4, "the core: 64-dimensional UNNAMED latent\nmemory smeared through the network\n~214 tuning knobs", RED, fc="#fdeeee")
box(ax, 0.4, 1.2, 9.2, 2.4, "interpretability AFTER training:\nprimitive discovery, K-means codes, counterfactual replay\n(a probe of the policy, not the policy)", AMBER, fc="#fdf6e8")

ax = axes[1]
ax.set_title("CRYSTAL-1 — legible BY CONSTRUCTION", fontsize=12, color=INK)
box(ax, 0.4, 7.6, 9.2, 1.6, "memory = NAMED belief on a K-simplex: P(bear), …\nthe sole memory channel (bottleneck, burst-blind TV = 0)", GREEN, fc="#eef7ee")
box(ax, 0.4, 4.4, 9.2, 2.4, "the policy head IS an ≤8-leaf soft tree\nat full return parity (gap −0.018, p≈0.84)\n— the story is the policy, not a probe of it", BLUE, fc="#eaf2fc")
box(ax, 0.4, 1.2, 9.2, 2.4, "5 named command levers (vs ~214 knobs):\nSET_BELIEF · EXPOSURE_MODE · CONCENTRATION_CAP\nABSTAIN_FLOOR · DRAWDOWN_BUDGET  + writ ladder & ledger", PURPLE, fc="#f1effa")
fig.tight_layout(); fig.savefig("figures/fig_architectures.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- F3: the influence levers
# Sources: E-27/E-28 (crystal_ppo.py runs), M6 (BC warm start), C-2 (dose), NI bar refusal.
fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.5))

ax = axes[0]  # R: reward shaping moves the drawdown dial
budgets = ["5%", "10%", "15%"]
realized = [4.81, 10.4, 15.5]  # dd05 head hold maxDD; R-lever middle dial; approx budget-15 (B&H-like)
bars = ax.bar(budgets, realized, color=[BLUE, BLUE, LIGHTBLUE], width=0.55)
for b, v in zip(bars, realized):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"−{v}%", ha="center", fontsize=10,
            fontweight="bold", color=INK)
ax.plot([-0.4, 2.4], [5, 5], ls=":", c=MUT, lw=1); ax.plot([-0.4, 2.4], [10, 10], ls=":", c=MUT, lw=1)
ax.plot([-0.4, 2.4], [15, 15], ls=":", c=MUT, lw=1)
ax.set_ylim(0, 18); ax.set_xlabel("drawdown budget in the reward")
ax.set_ylabel("realized max drawdown (hold)")
ax.set_title("R · reward shaping:\nthe budget is obeyed", fontsize=11)

ax = axes[1]  # T: teacher cures near-uniformity
cats = ["cold\nPPO", "BC\nteacher", "PPO\nfine-tune"]
vals = [0.02, 0.87, 0.87]
bars = ax.bar(cats, vals, color=[RED, BLUE, BLUE], width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10,
            fontweight="bold", color=INK)
ax.set_ylim(0, 1.05); ax.set_ylabel("agreement with the certified rule")
ax.set_title("T · teacher/BC:\ncures fake competence", fontsize=11)

ax = axes[2]  # I: interventions — write fidelity + honest refusal
bars = ax.bar(["write\nfidelity", "±2-nat\nslide", "+1-nat\n(NI bar)"],
              [1.0, 1.0, 0.0], color=[BLUE, BLUE, RED], width=0.55)
ax.text(bars[0].get_x() + 0.275, 1.02, "100%", ha="center", fontsize=10, fontweight="bold")
ax.text(bars[1].get_x() + 0.275, 1.02, "works", ha="center", fontsize=10, fontweight="bold")
ax.text(bars[2].get_x() + 0.275, 0.04, "REFUSED", ha="center", fontsize=10, fontweight="bold", color=RED)
ax.set_ylim(0, 1.15); ax.set_ylabel("intervention outcome")
ax.set_title("I · belief writes (SET_BELIEF):\nobeyed — and refusable", fontsize=11)

ax = axes[3]  # A: architecture depth = legibility dial
bars = ax.bar(["depth-2\n(4 leaves)", "depth-3\n(8 leaves)"], [4, 8], color=[BLUE, LIGHTBLUE], width=0.5)
for b, v in zip(bars, [4, 8]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v} leaves", ha="center", fontsize=10,
            fontweight="bold", color=INK)
ax.set_ylim(0, 9.5); ax.set_ylabel("story size (leaves)")
ax.set_title("A · architecture:\nlegibility as a dial", fontsize=11)

fig.tight_layout(); fig.savefig("figures/fig_levers.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- F4: the honest frontier
# Conceptual scatter grounded in measured points: CrystalScore vs where each policy lives,
# plus the certified G12 tension arrow (legibility is not free on hard substrates).
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.scatter([0.62], [0.151], s=180, c=LIGHTBLUE, edgecolors=MUT, zorder=3)
ax.annotate("R6c (CHRL line)\ndeployed, real panels", (0.62, 0.151), textcoords="offset points",
            xytext=(12, -4), fontsize=10, color=INK)
ax.scatter([0.60], [0.938], s=180, c=BLUE, edgecolors=MUT, zorder=3)
ax.annotate("CRYSTAL-1\npolygon-proven", (0.60, 0.938), textcoords="offset points",
            xytext=(12, -8), fontsize=10, color=INK)
ax.annotate("", xy=(0.88, 0.72), xytext=(0.72, 0.90),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
ax.text(0.905, 0.70, "the certified tension (G12):\non harder substrates, pushing return\npulls legibility DOWN — the open frontier",
        fontsize=9.5, color=RED, va="top")
ax.set_xlim(0.4, 1.35); ax.set_ylim(0, 1.05)
ax.set_xlabel("behavioral complexity of the substrate →")
ax.set_ylabel("CrystalScore")
ax.set_xticks([])
ax.set_title("The legibility frontier: where the two agents sit, and where the fight is", fontsize=12)
fig.tight_layout(); fig.savefig("figures/fig_frontier.png", dpi=150); plt.close(fig)

print("wrote 4 figures")
