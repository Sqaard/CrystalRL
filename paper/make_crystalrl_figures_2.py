# Figures F5-F6 for "CrystalRL" v0.3: the CHRL architecture panel and the Interpretable-CHRL
# program panel. Every number is copied verbatim from the two public repo READMEs
# (Sqaard/CHRL-Constrained-Hierarchical-Reinforcement-Learning, Sqaard/Interpretable-CHRL)
# and the Interpretable_CHRL_short_note.docx. Palette = the session-validated dataviz slots.
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


def box(ax, x, y, w, h, text, ec, fc="white", fs=9.5, lw=1.6):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


# ---------------------------------------------------------------- F5: CHRL architecture + validation
# Four-fold walk-forward vs the 100%-risky equal-weight universe benchmark (CHRL repo README):
# return 10.05 vs 17.09 (58.8% capture), Sharpe 1.09 vs 1.17 (93.2% capture), maxDD -7.34 vs
# -15.27 (52% cut); mean cash 41.24%, mean daily L1 turnover 0.84%.
fig = plt.figure(figsize=(12.6, 4.2))
gs = fig.add_gridspec(1, 4, width_ratios=[2.2, 0.75, 0.75, 0.75], wspace=0.5)
ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.set_title("three readable decision layers + five guardrails", fontsize=11.5, color=INK)
box(ax, 0.3, 7.4, 6.1, 1.9, "RISK layer (~20d rhythm)\ncash vs risky exposure", BLUE, fc="#eaf2fc")
box(ax, 0.3, 4.6, 6.1, 1.9, "GROUP layer\nroutes capital through stock clusters", BLUE, fc="#eaf2fc")
box(ax, 0.3, 1.8, 6.1, 1.9, "STOCK layer (~5d rhythm)\nbuy/sell candidate selection", BLUE, fc="#eaf2fc")
for y in (7.4, 4.6):
    ax.annotate("", xy=(3.35, y - 0.85), xytext=(3.35, y - 0.35),
                arrowprops=dict(arrowstyle="->", color=MUT, lw=1.4))
box(ax, 6.9, 1.8, 2.9, 7.5,
    "guardrails\n\nexposure pacing\nconfidence-scaled\nsizing\nevent triggers\ntop-K routing\nbuy gate",
    AMBER, fc="#fdf6e8", fs=9)
panels = [("return (%)", [10.05, 17.09], "58.8% capture"),
          ("Sharpe", [1.09, 1.17], "93.2% capture"),
          ("|max drawdown| (%)", [7.34, 15.27], "52% reduction")]
for i, (lab, vals, note) in enumerate(panels):
    ax = fig.add_subplot(gs[i + 1])
    bars = ax.bar(["CHRL", "bench"], vals, color=[BLUE, LIGHTBLUE], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:g}", ha="center", fontsize=9,
                fontweight="bold", color=INK)
    ax.set_title(lab, fontsize=10); ax.set_ylim(0, max(vals) * 1.25)
    ax.text(0.5, -0.26, note, transform=ax.transAxes, ha="center", fontsize=8.5, color=MUT)
fig.tight_layout(); fig.savefig("figures/fig_chrl.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- F6: the Interpretable-CHRL program
# Left: the staged pipeline over a FROZEN policy (short note + repo README). Right: counterfactual
# interventions on the frozen 2022-23 rollout, raw vs control-adjusted deltas (README verbatim):
# promote code4 +0.540/+0.263; replace code3->best +0.421/+0.302; hidden-action adapter
# +0.377/+0.265; targeted repairs +0.508 raw with random-repair control +0.316 -> adjusted +0.192.
# A random-target joint edit scores +0.552 - the reason control adjustment is mandatory.
fig = plt.figure(figsize=(12.6, 4.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.16)
ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.set_title("the staged program over a FROZEN policy", fontsize=11.5, color=INK)
stages = [
    ("log", "hidden states + action chunks from the frozen rollout", BLUE, "#eaf2fc"),
    ("discover", "K-means codebook: K=8, utilization 1.000,\nperplexity 6.91, median run 18d, cross-fold NMI 0.634", GREEN, "#eef7ee"),
    ("label + diagnose", "behavioral labels from trading logs;\noutcome diagnostics per primitive", BLUE, "#eaf2fc"),
    ("intervene", "suppress / promote / replace / repair:\none edited latent through the frozen action head", PURPLE, "#f1effa"),
    ("control", "matched-random twins + control-adjusted deltas\n(a random-target joint edit also 'improves': +0.552pp)", RED, "#fdeeee"),
]
y = 9.4
for name, text, ec, fc in stages:
    box(ax, 0.3, y - 1.6, 9.4, 1.6, name + ":  " + text, ec, fc=fc, fs=8.8)
    if y - 1.6 > 1.0:
        ax.annotate("", xy=(5.0, y - 1.94), xytext=(5.0, y - 1.66),
                    arrowprops=dict(arrowstyle="->", color=MUT, lw=1.2))
    y -= 1.95
ax2 = fig.add_subplot(gs[1])
labs = ["promote\ncode4", "replace\ncode3 to best", "hidden-action\nadapter", "joint + targeted\nrepairs"]
raw = [0.540, 0.421, 0.377, 0.508]
adj = [0.263, 0.302, 0.265, 0.192]
x = np.arange(4); w = 0.38
b1 = ax2.bar(x - w / 2, raw, w, color=LIGHTBLUE, label="raw counterfactual delta")
b2 = ax2.bar(x + w / 2, adj, w, color=BLUE, label="control-adjusted (vs matched random)")
for bars in (b1, b2):
    for b in bars:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                 "+" + format(b.get_height(), ".2f"), ha="center", fontsize=8.6,
                 fontweight="bold", color=INK)
ax2.axhline(0.552, ls=":", c=RED, lw=1.4)
ax2.text(0.02, 0.512, "random-target joint edit: +0.552", ha="left", fontsize=8.6, color=RED)
ax2.set_xticks(x); ax2.set_xticklabels(labs, fontsize=9)
ax2.set_ylabel("return delta, frozen 2022-23 rollout (pp)")
ax2.set_ylim(0, 0.80)
ax2.set_title("interventions survive matched-random controls,\nat a fraction of their raw size", fontsize=11)
ax2.legend(loc="upper center", frameon=False, fontsize=8.8, ncol=1, bbox_to_anchor=(0.62, 1.0))
fig.tight_layout(); fig.savefig("figures/fig_interp.png", dpi=150); plt.close(fig)

print("wrote figures/fig_chrl.png + figures/fig_interp.png")
