#!/usr/bin/env python3
"""Pipeline lane diagram — clean rows, shared base model node per model."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Palette ───────────────────────────────────────────────────────────────
C_BASE  = "#1E3A5F"
C_CPT   = "#0D7377"
C_SFT   = "#6B2D8B"
C_MIWV  = "#B8621A"
C_END   = "#1A5E3A"
C_LINE  = "#AAAAAA"
WHITE   = "#FFFFFF"
BG      = "#F8F9FA"

BW, BH = 1.6, 0.46
END_W  = 2.1

XB, XCPT, XSFT, XMWV, XEND = 1.3, 3.5, 5.8, 8.1, 11.0
ROW_GAP   = 0.82
GROUP_GAP = 0.95

fig, ax = plt.subplots(figsize=(14, 9.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9.5)
ax.axis("off")
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

def box(x, y, label, color, fs=8.3, w=BW, h=BH):
    r = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle="round,pad=0.07", linewidth=0,
                        facecolor=color, zorder=4)
    ax.add_patch(r)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fs, color=WHITE, fontweight="bold", zorder=5)

def end_box(x, y, label):
    r = FancyBboxPatch((x - END_W/2, y - BH/2), END_W, BH,
                        boxstyle="round,pad=0.07", linewidth=0,
                        facecolor=C_END, zorder=4)
    ax.add_patch(r)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=8.2, color=WHITE, fontweight="bold", zorder=5)

rows_8b = [
    ("8B  Baseline",           False, False, False),
    ("8B + SFT",               False, True,  False),
    ("8B + SFT + MIWV",        False, True,  True),
    ("8B + CPT + SFT",         True,  True,  False),
    ("8B + CPT + SFT + MIWV",  True,  True,  True),
]
rows_70b = [
    ("70B  Baseline",          False, False, False),
    ("70B + SFT",              False, True,  False),
    ("70B + SFT + MIWV",       False, True,  True),
]

def draw_section(rows, top_y, model_label):
    ys = [top_y - i * ROW_GAP for i in range(len(rows))]
    base_y = (ys[0] + ys[-1]) / 2

    # shared base model node
    r = FancyBboxPatch((XB - BW/2, base_y - 0.62), BW, 1.24,
                        boxstyle="round,pad=0.09", linewidth=0,
                        facecolor=C_BASE, zorder=4)
    ax.add_patch(r)
    ax.text(XB, base_y, f"Llama 3.1\n{model_label}", ha="center", va="center",
            fontsize=8.8, color=WHITE, fontweight="bold", zorder=5)

    vx = XB + BW/2 + 0.12   # x of the vertical spine

    for (label, use_cpt, use_sft, use_miwv), y in zip(rows, ys):
        # vertical spine branch to this row
        ax.plot([vx, vx], [base_y, y], color=C_LINE, lw=1.15, zorder=2)

        # continuous horizontal line from spine to end node
        line_end = XEND - END_W/2 - 0.06
        ax.plot([vx, line_end], [y, y], color=C_LINE, lw=1.15, zorder=2)

        # arrowhead just before end node
        ax.annotate("", xy=(line_end, y), xytext=(line_end - 0.12, y),
                    arrowprops=dict(arrowstyle="-|>", color=C_LINE,
                                    lw=1.15, mutation_scale=10), zorder=3)

        # place active step nodes on top of the line
        if use_cpt:
            box(XCPT, y, "CPT", C_CPT)
        if use_sft:
            box(XSFT, y, "SFT", C_SFT)
        if use_miwv:
            box(XMWV, y, "MIWV", C_MIWV)

        end_box(XEND, y, label)

    return ys[0], ys[-1]

# ── 8B section ────────────────────────────────────────────────────────────
top_8b = 8.8
t8, b8 = draw_section(rows_8b, top_8b, "8B")

# ── Divider ───────────────────────────────────────────────────────────────
div_y = b8 - GROUP_GAP / 2
ax.plot([0.3, 13.7], [div_y, div_y], color="#DDDDDD", lw=0.9, ls="--")

# ── 70B section ───────────────────────────────────────────────────────────
top_70b = div_y - GROUP_GAP / 2
t70, b70 = draw_section(rows_70b, top_70b, "70B")

ax.text(XCPT, b70 - 0.45,
        "CPT not applied — est. 80–100 GPU hrs,\ncorpus too small to avoid overfitting",
        ha="center", va="center", fontsize=7.2, color="#AAAAAA", style="italic")

# ── Column headers ────────────────────────────────────────────────────────
HDR_Y = top_8b + 0.45
for x, lbl, col in [
    (XB,   "Base Model",  C_BASE),
    (XCPT, "CPT",         C_CPT),
    (XSFT, "SFT",         C_SFT),
    (XMWV, "MIWV",        C_MIWV),
    (XEND, "Final Model", C_END),
]:
    ax.text(x, HDR_Y, lbl, ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=col)

ax.plot([0.3, 13.7], [HDR_Y - 0.08, HDR_Y - 0.08], color="#CCCCCC", lw=0.8)

# ── Abbreviation key ──────────────────────────────────────────────────────
abbrevs = [
    (C_CPT,  "CPT  —  Continued Pre-Training"),
    (C_SFT,  "SFT  —  Supervised Fine-Tuning (instruction tuning)"),
    (C_MIWV, "MIWV —  Model Instruction Weakness Value (data selection, top 10%)"),
]
kx, ky = 0.4, 0.62
for color, label in abbrevs:
    dot = FancyBboxPatch((kx, ky - 0.11), 0.2, 0.22,
                          boxstyle="round,pad=0.02", linewidth=0,
                          facecolor=color, zorder=5)
    ax.add_patch(dot)
    ax.text(kx + 0.3, ky, label, va="center", fontsize=7.4, color="#444444", zorder=5)
    ky += 0.44

plt.tight_layout(pad=0.4)
plt.savefig("/home/skavlak/finetuning/figures/experimental_conditions.png",
            dpi=200, bbox_inches="tight", facecolor=BG)
plt.savefig("/home/skavlak/finetuning/figures/experimental_conditions.pdf",
            bbox_inches="tight", facecolor=BG)
print("Saved.")
