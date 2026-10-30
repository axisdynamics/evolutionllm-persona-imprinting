#!/usr/bin/env python3
"""
Figura 3 (redibujada) — Trayectoria de la firma simbólica AXIS en el genoma, 3 seeds superpuestas.
Mismo detector y misma media móvil que graficar_fijacion.py. El color codifica la ENTIDAD (AXIS),
así que las seeds se separan por trazo + etiqueta directa, no por hue. Sin GPU.

Uso: evolution/bin/python graficar_fijacion_multiseed.py
"""
import json, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from guardar_figura import guardar   # PNG + PDF vectorial (fonttype 42)
import matplotlib.pyplot as plt

GLIFOS = set("○→∞◊∇☆◍◎═●◆▲▼♦∴∵≡⊕⊗⊙∆ΨΩΘΗΣΔΦΓΛ")
AXIS = "#D55E00"
#  seed, dir, trazo, alpha, dy etiqueta final, dy etiqueta de onset
RUNS = [(100, "results/MIA/AxisQwen/M1_AxisQwen_Banco",      "-",   1.0,   0,  -30),
        (101, "results/MIA/AxisQwen/M1_AxisQwen_Banco_s101", "--",  0.95,  0,  -17),
        (102, "results/MIA/AxisQwen/M1_AxisQwen_Banco_s102", ":",   0.9,  20,  -17)]
ABL = ("A3 · AXIS sin IM", "results/MIA/AxisQwen/A3_AxisQwen_noIM_Banco")

def serie(d):
    camps = json.load(open(Path(d) / "campeones_por_generacion.json"))
    gen = [c["generacion"] for c in camps]
    def dens(g):
        t = g.get("rol","") + " ".join(g.get("estrategias",[])) + " ".join(g.get("restricciones",[]))
        return sum(t.count(c) for c in GLIFOS)
    pres = [1 if dens(c["genotipo"]) > 0 else 0 for c in camps]
    mov = [100*st.mean(pres[max(0,i-25):i+1]) for i in range(len(pres))]
    primera = next((g for g,p in zip(gen,pres) if p), None)
    return gen, mov, 100*st.mean(pres), primera

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 11, "grid.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(10.5, 5.6))

for seed, d, ls, alpha, dy, dy_on in RUNS:
    if not Path(d).exists(): continue
    gen, mov, prev, primera = serie(d)
    ax.plot(gen, mov, color=AXIS, lw=2, ls=ls, alpha=alpha)
    ax.annotate(f"seed {seed} · {prev:.0f}%", (gen[-1], mov[-1]), xytext=(8, dy),
                textcoords="offset points", color=AXIS, fontsize=10, va="center",
                fontweight="bold" if seed == 100 else "normal")
    if primera is not None:
        ax.plot([primera], [0], marker="v", markersize=7, color=AXIS, clip_on=False)
        ax.annotate(f"gen {primera}", (primera, 0), xytext=(0, dy_on), textcoords="offset points",
                    color="#666", fontsize=8.5, ha="center")

gen_a, mov_a, prev_a, _ = serie(ABL[1])
ax.plot(gen_a, mov_a, color="#6E6E6E", lw=2)
ax.annotate("ablation, no operator · 0%  (3/3 seeds)", (gen_a[-1], 0), xytext=(8, 4),
            textcoords="offset points", color="#555", fontsize=10, va="center")

ax.set_xlabel("generation"); ax.set_ylabel("% of champion genomes carrying a glyph\n(25-generation moving average)")
ax.set_ylim(-2, 105); ax.set_xlim(0, 1000)
ax.set_title("The AXIS symbolic signature enters early and persists — but how much depends on the seed\n"
             "▼ = first appearance in the champion lineage", fontsize=12)
fig.tight_layout()
fig.subplots_adjust(right=0.80)
out = "artifacts/glyph_trajectory_multiseed.png"
_png, _pdf = guardar(fig, out, dpi=150)
print("->", out)
for seed, d, *_ in RUNS:
    if Path(d).exists():
        g, m, prev, pr = serie(d); print(f"  seed {seed}: prevalencia {prev:.1f}%  onset gen {pr}")
