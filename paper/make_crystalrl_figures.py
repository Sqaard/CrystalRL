# Figures for "CrystalRL — Crystal Clear Reinforcement Learning" (v0.2: real-data evidence).
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

# ---------------------------------------------------------------- F1: CrystalScore, REAL data
# R6c axes: its real stance log (the deployed panel run) — CRYSTAL1_CSI500_AND_CRYSTALSCORE.md.
# CRYSTAL-1 axes on the real US machinery: Simulatability = the 8-leaf tree reproducing the DP
# champion on the US universe (W4, personal_invest_dp_report.json, 0.92); Stability = 10-seed
# behavioral stability of the certified Dow rule (E-21 audit, 1.0); Faithfulness = belief-write
# fidelity on the real daily panel (E-28, 1.0).
fig, ax = plt.subplots(figsize=(8.6, 4.0))
axes_names = ["Faithfulness", "Simulatability", "Stability", "CrystalScore\n(product, K≤9)"]
r6c = [1.00, 0.244, 0.619, 0.151]
c1 = [1.00, 0.92, 1.00, 0.92]
x = np.arange(4); w = 0.36
b1 = ax.bar(x - w / 2, r6c, w, color=LIGHTBLUE, label="R6c (CHRL line): its deployed-panel stance log")
b2 = ax.bar(x + w / 2, c1, w, color=BLUE, label="CRYSTAL-1: US goal machinery + certified Dow rule")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
ax.set_xticks(x); ax.set_xticklabels(axes_names)
ax.set_ylim(0, 1.14); ax.set_ylabel("score (identical scalar, real data)")
ax.set_title("Born-legible vs explained-later, on real market data: the gap is Simulatability",
             fontsize=12)
ax.legend(loc="center left", frameon=False, fontsize=9.5)
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
box(ax, 0.4, 7.6, 9.2, 1.6, "memory = NAMED belief on a K-simplex: P(bear), …\nthe sole memory channel (a Bayes filter over regimes)", GREEN, fc="#eef7ee")
box(ax, 0.4, 4.4, 9.2, 2.4, "the policy head IS a small decision tree / readable rule\nat full performance parity\n— the story is the policy, not a probe of it", BLUE, fc="#eaf2fc")
box(ax, 0.4, 1.2, 9.2, 2.4, "5 named command levers (vs ~214 knobs):\nSET_BELIEF · EXPOSURE_MODE · CONCENTRATION_CAP\nABSTAIN_FLOOR · DRAWDOWN_BUDGET  + writ ladder & ledger", PURPLE, fc="#f1effa")
fig.tight_layout(); fig.savefig("figures/fig_architectures.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- F3: the influence levers (real daily panel)
# Sources: E-27/E-28 (crystal_ppo.py on the real US daily panel, train 2010-18 / hold),
# M6 warm-start lesson, NI-bar refusal.
fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.5))

ax = axes[0]  # R: the reward penalty holds only the 5% budget; 8%/12% breach
budgets = ["5%", "8%", "12%"]
realized = [4.81, 15.9, 15.9]
obeys = [True, False, False]
bars = ax.bar(budgets, realized, color=[GREEN, RED, RED], width=0.55)
for b, v, ok in zip(bars, realized, obeys):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"−{v}%\n{'holds' if ok else 'BREACH'}",
            ha="center", fontsize=9, fontweight="bold", color=GREEN if ok else RED)
for bl, x0, x1 in [(5, -0.4, 0.4), (8, 0.6, 1.4), (12, 1.6, 2.4)]:
    ax.plot([x0, x1], [bl, bl], ls=":", c=MUT, lw=1.2)
ax.set_ylim(0, 20); ax.set_xlabel("drawdown budget written into the reward")
ax.set_ylabel("realized max drawdown (hold)")
ax.set_title("R · reward penalty:\nbudget NOT enforced (only 5%)", fontsize=11)

ax = axes[1]  # T: teacher warm-start cures the cold near-uniform collapse
cats = ["cold\nPPO", "teacher\nwarm"]
vals = [0.21, 0.73]
bars = ax.bar(cats, vals, color=[RED, GREEN], width=0.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10,
            fontweight="bold", color=INK)
ax.axhline(0.20, ls=":", c=MUT, lw=1)
ax.text(1.42, 0.22, "1/5 = near-uniform", ha="right", fontsize=8, color=MUT)
ax.set_ylim(0, 0.85); ax.set_ylabel("mean max action-probability")
ax.set_title("T · teacher warm-start:\ncures the near-uniform collapse", fontsize=11)

ax = axes[2]  # I: the control battery — the intervention is an artifact; only the refusal survives
labs = ["defensive\nleaf", "matched-\nrandom", "NI\nrefusal"]
vals = [0.50, 0.50, 1.0]
cols = [AMBER, AMBER, GREEN]
bars = ax.bar(labs, vals, color=cols, width=0.6)
ax.text(bars[0].get_x() + 0.3, 0.53, "moves", ha="center", fontsize=9, fontweight="bold", color=INK)
ax.text(bars[1].get_x() + 0.3, 0.53, "= moves\n(placebo\nfails)", ha="center", fontsize=8,
        fontweight="bold", color=RED)
ax.text(bars[2].get_x() + 0.3, 1.02, "REFUSED", ha="center", fontsize=9, fontweight="bold", color=GREEN)
ax.set_ylim(0, 1.2); ax.set_ylabel("|Δ mean exposure| (cold head)")
ax.set_title("I · intervention under controls:\nartifact — only refusal survives", fontsize=11)

ax = axes[3]  # A: the head's CrystalScore — simulatable + stable, but not faithful
labs = ["Faith-\nfulness", "Simulat-\nability", "Stab-\nility", "Crystal\nScore"]
vals = [0.03, 1.0, 1.0, 0.03]
cols = [RED, GREEN, GREEN, RED]
bars = ax.bar(labs, vals, color=cols, width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9,
            fontweight="bold", color=INK)
ax.set_ylim(0, 1.15); ax.set_ylabel("score")
ax.set_title("A · the head's CrystalScore:\nlegible dial, not faithful", fontsize=11)

fig.tight_layout(); fig.savefig("figures/fig_levers.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- F4: the designed-market experiment
# The VoI gate: value of the belief information, designed market vs a real book
# (CRYSTAL1_CONTROLLABILITY_FINAL_REPORT.md, debt 4: polygon VoI 5.92 = +282% of blind; real
# crypto book VoI 0 -> the gate CLOSES).
fig, ax = plt.subplots(figsize=(7.4, 3.8))
bars = ax.bar(["designed market\n(belief value planted\nby construction)", "real crypto book\n(the regime is priced)"],
              [5.92, 0.0], color=[BLUE, LIGHTBLUE], width=0.45)
ax.text(bars[0].get_x() + 0.225, 6.02, "VoI = 5.92  (+282% of blind)\n→ gate OPEN", ha="center",
        fontsize=10, fontweight="bold", color=INK)
ax.text(bars[1].get_x() + 0.225, 0.25, "VoI = 0 → gate CLOSED", ha="center", fontsize=10,
        fontweight="bold", color=RED)
ax.set_ylim(0, 7.6); ax.set_ylabel("value of the belief information (VoI)")
ax.set_title("The designed-market experiment: a market that rewards only a clean mind —\nand the gate that keeps its results out of real-market claims", fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig_polygon.png", dpi=150); plt.close(fig)

print("wrote 4 figures (crystalscore/architectures/levers/polygon)")
