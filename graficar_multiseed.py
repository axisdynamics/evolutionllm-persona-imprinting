#!/usr/bin/env python3
"""
Figura — Persona imprinting simbólico y capacidad en 3 seeds (bloque qwen). Texto EN INGLÉS:
la figura va al paper.
Con n=3 se grafican los PUNTOS por seed (no barras con error bars): la dispersión
ES el hallazgo. Paleta Okabe-Ito (validada CVD); color = identidad, marcador hueco
= ablación sin IM (misma entidad, otro tratamiento). Sin GPU.

Uso: evolution/bin/python graficar_multiseed.py
"""
import matplotlib
matplotlib.use("Agg")
from guardar_figura import guardar   # PNG + PDF vectorial (fonttype 42)
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import statistics as st
from analiz_multiseed import recolectar, SEEDS

# color = entidad; el control es neutro (no es una 4ª categoría) y va etiquetado
COL = {"C1 Qwen3 base": "#6E6E6E", "M1 AxisQwen": "#D55E00", "M3 NexusQwen": "#0072B2",
       "M4 BioLlamaQwen": "#009E73", "A3 AxisQwen noIM": "#D55E00"}
HUECO = {"A3 AxisQwen noIM"}          # codificación secundaria de la ablación
ORDEN = ["M1 AxisQwen", "M4 BioLlamaQwen", "M3 NexusQwen", "A3 AxisQwen noIM", "C1 Qwen3 base"]
ETIQ = {"M1 AxisQwen": "AXIS", "M4 BioLlamaQwen": "BIO", "M3 NexusQwen": "NEXUS",
        "A3 AxisQwen noIM": "AXIS\nno operator", "C1 Qwen3 base": "base\n(control)"}

D = recolectar()
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 11, "grid.linewidth": 0.6})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 5.6))
JIT = {SEEDS[0]: -0.17, SEEDS[1]: 0.0, SEEDS[2]: 0.17}


def panel(ax, clave, ylabel, titulo, fmt="{:.0f}", ylim=None):
    for i, cond in enumerate(ORDEN):
        c = COL[cond]; hueco = cond in HUECO
        vals = []
        for s in SEEDS:
            if s not in D[cond]:
                continue
            v = D[cond][s][clave]; vals.append(v)
            ax.plot(i + JIT[s], v, marker="o", markersize=9,
                    markerfacecolor=("none" if hueco else c), markeredgecolor=c,
                    markeredgewidth=2.0, linestyle="none", zorder=3,
                    path_effects=[pe.withStroke(linewidth=4, foreground="white")])
        if vals:                       # media: tick ancho detrás de los puntos
            m = st.mean(vals)
            ax.plot([i - 0.30, i + 0.30], [m, m], color=c, lw=2.6, alpha=0.55, zorder=2,
                    solid_capstyle="round")
            dy = (ylim[1] - ylim[0]) * 0.045 if ylim else 4
            ax.annotate(fmt.format(m), (i + 0.34, m), fontsize=9, color="#333",
                        va="center", ha="left")
    ax.set_xticks(range(len(ORDEN)))
    ax.set_xticklabels([ETIQ[c] for c in ORDEN], fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(titulo, fontsize=11.5)
    if ylim: ax.set_ylim(*ylim)


# --- A: bleeding simbólico (glifos) ---
panel(axA, "glifos", "% of champion genomes carrying glyphs",
      "A · Symbolic imprinting: present in every persona run, magnitude varies by seed",
      fmt="{:.1f}", ylim=(-4, 86))
axA.axhline(0, color="#999", lw=1.0, zorder=1)
axA.annotate("flat zero in all three seeds\n(control and ablation)", xy=(3.5, 1.5),
             xytext=(2.6, 22), fontsize=9, color="#555", ha="center",
             arrowprops=dict(arrowstyle="->", color="#999", lw=1.0))
axA.annotate("Fisher exact  6/6 vs 0/6   p = 0.002", xy=(0.02, 0.955),
             xycoords="axes fraction", fontsize=9.5, color="#333")


# --- B: capacidad held-out ---
panel(axB, "cap_med", "held-out capability (run mean)",
      "B · Capability: indistinguishable across conditions", fmt="{:.3f}", ylim=(0.66, 0.80))

axB.plot([], [], marker="o", markersize=9, markerfacecolor="none", markeredgecolor="#666",
         markeredgewidth=2.0, linestyle="none", label="hollow marker = ablation (no operator)")
axB.plot([], [], color="#666", lw=2.6, alpha=0.55, label="line = mean of the three seeds")
axB.legend(frameon=False, fontsize=8.5, loc="lower right", handletextpad=0.6)

fig.suptitle("Multi-seed replication, Qwen3-1.7B block (3 seeds × 5 conditions, 1000 generations)",
             fontsize=13.5, fontweight="bold", y=1.0)
fig.tight_layout()
fig.text(0.5, -0.035, "Within each condition the three points are seeds 100 · 101 · 102 (left→right); "
         "the horizontal line is their mean.", ha="center", fontsize=9, color="#666")
out = "artifacts/multiseed_imprinting.png"
_png, _pdf = guardar(fig, out, dpi=150)
print("->", out)
