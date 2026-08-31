"""Three levels of causal integration (conceptual, no data).

Left:   transparent layers   x -> h -> a, constraints around h
Middle: post-hoc             x -> h -> a, explanation h -> b fitted outside the policy
Right:  born legible         x -> b -> pi_leg -> a, named belief inside the executed path

Run:  python paper/figures/make_fig_generations.py
Out:  paper/figures/fig_generations.pdf (vector)
      paper/figures/fig_generations.png (300-dpi Word-compatible preview)
"""
from pathlib import Path
from datetime import datetime, timezone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "figure.dpi": 140,
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT_DIR = Path(__file__).resolve().parent
OUT_PDF = OUT_DIR / "fig_generations.pdf"
OUT_PNG = OUT_DIR / "fig_generations.png"

OPAQUE = "#5B6573"
POSTHOC = "#E69F00"
BORN = "#0072B2"
SUPPORTED = "#009E73"
CONTROL = "#B8BDC5"
EDGE = "#26313D"
PDF_METADATA = {
    "Creator": "CrystalRL figure script",
    "Producer": "Matplotlib",
    "CreationDate": datetime(2026, 8, 31, tzinfo=timezone.utc),
}


def box(ax, x, y, w, h, text, fc, ls="-", fs=9.5, weight="normal", tc=EDGE,
        hatch=None, shape="round"):
    boxstyle = "round,pad=0.02,rounding_size=0.04" if shape == "round" else "square,pad=0.02"
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle=boxstyle, fc=fc, ec=EDGE, lw=1.1,
                                ls=ls, hatch=hatch))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, weight=weight, color=tc)


def arrow(ax, p, q, ls="-", color=EDGE):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12,
                                 lw=1.2, color=color, ls=ls, shrinkA=2, shrinkB=2))


def panel(ax, title, subtitle):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.97, title, ha="center", va="top", fontsize=11, weight="bold")
    ax.text(0.5, 0.905, subtitle, ha="center", va="top", fontsize=8.5, style="italic", color="#555")


fig, axes = plt.subplots(1, 3, figsize=(11, 4.6))

# ---- Panel 1: transparent layers, opaque core --------------------------------
ax = axes[0]
panel(ax, "transparent layers", "interpretability sits around the core")
box(ax, 0.5, 0.78, 0.30, 0.10, r"$x_t$", "white")
box(ax, 0.5, 0.56, 0.42, 0.12, r"$h_t$  (unnamed latent)", OPAQUE, tc="white")
box(ax, 0.5, 0.34, 0.30, 0.10, r"$a_t$", "white")
arrow(ax, (0.5, 0.73), (0.5, 0.62))
arrow(ax, (0.5, 0.50), (0.5, 0.39))
# constraints flanking the core
box(ax, 0.13, 0.56, 0.20, 0.28, "hier-\narchy", CONTROL, fs=8.5, hatch="///", shape="square")
box(ax, 0.87, 0.56, 0.20, 0.28, "hard\nrules", CONTROL, fs=8.5, hatch="///", shape="square")
ax.text(0.5, 0.02, "executed path: $x \\to h \\to a$\nconstraints govern $a$; the mechanism\nfrom $h$ to $a$ stays unnamed",
        ha="center", va="bottom", fontsize=8, color="#444")

# ---- Panel 2: post-hoc ----------------------------------------------------------
ax = axes[1]
panel(ax, "post-hoc explanation", "interpretability fitted outside the policy")
box(ax, 0.36, 0.78, 0.26, 0.10, r"$x_t$", "white")
box(ax, 0.36, 0.56, 0.36, 0.12, r"$h_t$  (frozen)", OPAQUE, tc="white")
box(ax, 0.36, 0.34, 0.26, 0.10, r"$a_t$", "white")
arrow(ax, (0.36, 0.73), (0.36, 0.62))
arrow(ax, (0.36, 0.50), (0.36, 0.39))
box(ax, 0.82, 0.56, 0.28, 0.12, r"$b_t = q_\phi(h_t)$", POSTHOC, ls="--")
arrow(ax, (0.545, 0.56), (0.675, 0.56), ls="--", color=POSTHOC)
box(ax, 0.82, 0.34, 0.26, 0.10, r"$\hat a_t = \hat\pi(b_t)$", POSTHOC, ls="--", fs=9)
arrow(ax, (0.82, 0.50), (0.82, 0.39), ls="--", color=POSTHOC)
ax.text(0.5, 0.02, "executed path unchanged; $q_\\phi$ read off $h$\nafter training — describes, does not\nguarantee $\\mathrm{do}(b)$ semantics",
        ha="center", va="bottom", fontsize=8, color="#444")

# ---- Panel 3: born legible -------------------------------------------------------
ax = axes[2]
panel(ax, "born legible", "interpretability inside the executed path")
box(ax, 0.5, 0.78, 0.30, 0.09, r"$x_t$", "white")
box(ax, 0.5, 0.62, 0.44, 0.11, r"$b_t = B(x_{\leq t})$  (named)", BORN, tc="white")
box(ax, 0.5, 0.46, 0.44, 0.11, r"$\pi_{\mathrm{leg}}(b_t, g_t)$  (small)", BORN, tc="white")
box(ax, 0.5, 0.30, 0.30, 0.09, r"$a_t$", "white")
arrow(ax, (0.5, 0.735), (0.5, 0.675))
arrow(ax, (0.5, 0.565), (0.5, 0.515))
arrow(ax, (0.5, 0.405), (0.5, 0.345))
# write surface
ax.text(0.90, 0.62, r"$\operatorname{do}(b_t=b^{*})$", ha="center", va="center",
        fontsize=8.5, color=SUPPORTED)
arrow(ax, (0.84, 0.62), (0.73, 0.62), color=SUPPORTED)
ax.text(0.5, 0.02, "executed path: $x \\to b \\to \\pi_{\\mathrm{leg}} \\to a$\nthe object interpreted is the object executed",
        ha="center", va="bottom", fontsize=8, color="#444")

fig.suptitle("Where the interpretable object sits relative to the executed decision path",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight", transparent=True, metadata=PDF_METADATA)
fig.savefig(OUT_PNG, bbox_inches="tight", transparent=True, dpi=300)
print(f"wrote {OUT_PDF}")
print(f"wrote {OUT_PNG}")
