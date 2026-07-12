# Build the CrystalRL live-testing page: generate the two model schemas (matplotlib),
# base64-embed them into site_template.html, write deck_assets/CrystalRL_live_testing.html.
# The result is ONE self-contained file — works from file://, live_server.py, or GitHub clone.
import base64
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2a78d6"; GREEN = "#008300"; AMBER = "#eda100"; PURPLE = "#4a3aa7"
RED = "#e34948"; SURF = "#ffffff"; INK = "#222222"; MUT = "#666666"

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "font.size": 10})


def box(ax, x, y, w, h, text, ec, fc="white", fs=9, lw=1.5):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=MUT, lw=1.3))


def r6c_schema():
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    box(ax, 0.3, 8.3, 9.4, 1.3, "market panel (prices, volumes, macro features)", MUT)
    arrow(ax, 5, 8.3, 5, 7.5)
    box(ax, 0.3, 5.4, 9.4, 2.1,
        "64-d UNNAMED latent (PPO core)\nmemory smeared through the network", RED, fc="#fdeeee")
    arrow(ax, 5, 5.4, 5, 4.6)
    box(ax, 0.3, 2.6, 9.4, 2.0,
        "3 readable control layers + guardrails:\nrisk (20d) → groups → stocks (5d);"
        " pacing · confidence scaling\nbuy gate · event de-risk · top-K routing", BLUE, fc="#eaf2fc")
    arrow(ax, 7.4, 2.6, 7.4, 1.8)
    box(ax, 5.2, 0.4, 4.5, 1.4, "portfolio weights", GREEN, fc="#eef7ee")
    ax.text(0.3, 1.1, "steering = ~214 engineering\nknobs (no named commands,\nno likelihood)",
            fontsize=8, color=RED, va="center")
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=110)
    plt.close(fig); return base64.b64encode(buf.getvalue()).decode()


def c1_schema():
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    box(ax, 0.3, 8.3, 9.4, 1.3, "market returns (observations)", MUT)
    arrow(ax, 5, 8.3, 5, 7.5)
    box(ax, 0.3, 5.4, 9.4, 2.1,
        "NAMED belief on a K-simplex:  P(bear), P(bull)\nBayes filter; the SOLE memory channel\n"
        "writable:  SET_BELIEF(bear)=0.8  is a command", GREEN, fc="#eef7ee")
    arrow(ax, 5, 5.4, 5, 4.6)
    box(ax, 0.3, 2.6, 9.4, 2.0,
        "the policy IS an ≤8-leaf soft tree over\n[belief, inventory, time] — at full return parity\n"
        "(the story is the policy, not a probe)", BLUE, fc="#eaf2fc")
    arrow(ax, 7.4, 2.6, 7.4, 1.8)
    box(ax, 5.2, 0.4, 4.5, 1.4, "exposure decision", GREEN, fc="#eef7ee")
    ax.text(0.3, 1.1, "steering = 5 named levers\nunder a writ ladder + ledger\n(certified writes)",
            fontsize=8, color=PURPLE, va="center")
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=110)
    plt.close(fig); return base64.b64encode(buf.getvalue()).decode()


def main():
    tpl = open("site_template.html", encoding="utf-8").read()
    out = tpl.replace("{{R6C_IMG}}", r6c_schema()).replace("{{C1_IMG}}", c1_schema())
    with open("deck_assets/CrystalRL_live_testing.html", "w", encoding="utf-8") as f:
        f.write(out)
    print("wrote deck_assets/CrystalRL_live_testing.html,", len(out) // 1024, "KB")


if __name__ == "__main__":
    main()
