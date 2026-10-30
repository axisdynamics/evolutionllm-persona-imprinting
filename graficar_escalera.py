#!/usr/bin/env python3
"""
Figura — Escalera de escala qwen3 (0.6B / 1.7B / 8B). Texto EN INGLÉS: va al paper.

Tres paneles, porque la historia tiene tres partes y ninguna se entiende sola:
  A) impronta SIMBÓLICA: U invertida — pica en 1.7B y desaparece en los dos extremos.
  B) impronta LÉXICA, como razón persona/base del MISMO substrato (eje log, referencia en 1×):
     también se apaga a 8B. Comparar contra el base de otro substrato fue el error que
     hizo creer que a 8B "la persona seguía ahí".
  C) capacidad: sube monótona con la escala y la persona nunca se separa de su base.

Uso: evolution/bin/python graficar_escalera.py
"""
import json, re, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from guardar_figura import guardar   # PNG + PDF vectorial (fonttype 42)
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from estilo import GLIFOS, LEXICONES

AXIS_COL, BASE_COL = "#D55E00", "#6E6E6E"
MF = "axis_qwen_3.1.txt"          # las cadenas base64 son idénticas en los 3 tamaños
ESCALA = [
    (0.6, "0.6B", "results/MIA/AxisQwen06b/E06_AxisQwen06b",
                  "results/Modelo Base/Qwen3 0.6B/Cb06_Qwen3_06B_s100"),
    (1.7, "1.7B", "results/MIA/AxisQwen/M1_AxisQwen_Banco",
                  "results/Modelo Base/Qwen3 1.7B/C1_Qwen3_Banco"),
    (8.0, "8B",   "results/MIA/AxisQwen8b/E8_AxisQwen8b",
                  "results/Modelo Base/Qwen3 8B/Cb8_Qwen3_8B_s100"),
]
SEEDS = [100, 101, 102]
ADN = re.findall(r"[A-Za-z0-9+/]{20,}={0,2}", open(MF, encoding="utf-8").read())

def geno(g):
    p=[g.get("rol",""),g.get("formato_salida","")]+(g.get("estrategias") or [])+(g.get("restricciones") or [])
    return " \n ".join(str(x) for x in p)

def medir(d):
    camps=json.load(open(f"{d}/campeones_por_generacion.json")); ng=lex=0
    for c in camps:
        t=geno(c["genotipo"])
        if sum(t.count(x) for x in GLIFOS)>0: ng+=1
        lex+=sum(t.lower().count(s) for s in LEXICONES["axis"])
    term=[json.loads(l) for l in open(f"{d}/termometro.jsonl") if l.strip()]
    return 100*ng/len(camps), lex, st.mean([t["fitness_heldout"] for t in term])

def ruta(base,s):
    if "M1_AxisQwen_Banco" in base: return base if s==100 else f"{base}_s{s}"
    return f"{base}_s{s}"

D={}
for x,lbl,base,ctrl in ESCALA:
    filas=[medir(ruta(base,s)) for s in SEEDS if Path(f"{ruta(base,s)}/campeones_por_generacion.json").exists()]
    D[lbl]=dict(x=x, gl=[f[0] for f in filas], lex=[f[1] for f in filas], cap=[f[2] for f in filas],
                base=medir(ctrl))

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 11, "grid.linewidth": 0.6})
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.5, 5.2))
XS=[d["x"] for d in D.values()]; JIT=[0.93,1.0,1.07]
anillo=[pe.withStroke(linewidth=4, foreground="white")]

def puntos(ax, ys, x, col, hueco=False):
    for j,v in enumerate(ys):
        ax.plot(x*JIT[j], v, marker="o", markersize=9, markeredgewidth=2.0,
                markerfacecolor=("none" if hueco else col), markeredgecolor=col,
                linestyle="none", zorder=4, path_effects=anillo)

# --- A: impronta simbólica ---
med=[st.mean(D[l]["gl"]) for _,l,_,_ in ESCALA]
a1.plot(XS, med, color=AXIS_COL, lw=2, zorder=3, alpha=.8)
for _,l,_,_ in ESCALA: puntos(a1, D[l]["gl"], D[l]["x"], AXIS_COL)
a1.plot(XS, [D[l]["base"][0] for _,l,_,_ in ESCALA], color=BASE_COL, lw=2, ls="--", zorder=2)
a1.annotate("base control", (8.0, 0), xytext=(-6, 10), textcoords="offset points",
            color=BASE_COL, fontsize=9, ha="right")
a1.set_ylabel("% of champion genomes carrying glyphs"); a1.set_ylim(-4, 86)
a1.set_title("A · Symbolic imprinting peaks in the middle", fontsize=11.5)
a1.annotate("38.8%", (1.7, 38.8), xytext=(10, 12), textcoords="offset points",
            fontsize=10, fontweight="bold", color=AXIS_COL)
for x,l in ((0.6,"0.3%"), (8.0,"0.0%")):
    a1.annotate(l, (x, 0), xytext=(0, 16), textcoords="offset points", fontsize=9.5,
                color=AXIS_COL, ha="center", fontweight="bold")

# --- B: impronta léxica como razón persona/base (mismo substrato) ---
for _,l,_,_ in ESCALA:
    b=D[l]["base"][1]
    puntos(a2, [v/b for v in D[l]["lex"]], D[l]["x"], AXIS_COL)
a2.plot(XS, [st.mean(D[l]["lex"])/D[l]["base"][1] for _,l,_,_ in ESCALA],
        color=AXIS_COL, lw=2, alpha=.8, zorder=3)
a2.axhline(1.0, color=BASE_COL, lw=2, ls="--", zorder=2)
a2.annotate("1× = no imprint above base", (0.55, 1.0), xytext=(0, -16),
            textcoords="offset points", fontsize=9, color=BASE_COL)
a2.set_yscale("log"); a2.set_ylabel("persona lexicon ÷ its own base control  (log)")
a2.set_title("B · The lexical channel switches off too", fontsize=11.5)
a2.annotate("×87", (0.6, 87), xytext=(8, 4), textcoords="offset points", fontsize=10,
            fontweight="bold", color=AXIS_COL)
a2.annotate("×0.9", (8.0, 0.94), xytext=(-4, -20), textcoords="offset points", fontsize=10,
            fontweight="bold", color=AXIS_COL, ha="right")

# --- C: capacidad ---
for _,l,_,_ in ESCALA: puntos(a3, D[l]["cap"], D[l]["x"], AXIS_COL)
a3.plot(XS, [st.mean(D[l]["cap"]) for _,l,_,_ in ESCALA], color=AXIS_COL, lw=2, alpha=.8, zorder=3,
        label="AXIS persona")
a3.plot(XS, [D[l]["base"][2] for _,l,_,_ in ESCALA], color=BASE_COL, lw=2, ls="--", zorder=2,
        label="base control")
a3.set_ylabel("held-out capability (run mean)")
a3.set_title("C · Capability rises; persona never departs from base", fontsize=11.5)
a3.legend(frameon=False, fontsize=9, loc="lower right")

for ax in (a1, a2, a3):
    ax.set_xscale("log"); ax.set_xticks(XS); ax.set_xticklabels([l for _,l,_,_ in ESCALA])
    ax.set_xlim(0.45, 11); ax.set_xlabel("substrate size (qwen3 family, log scale)")
    ax.minorticks_off()

fig.suptitle("Same persona, same family, three sizes: symbolic imprinting needs a competence window",
             fontsize=13.5, fontweight="bold", y=1.0)
fig.tight_layout()
fig.text(0.5, -0.03, "Three seeds per size (100 · 101 · 102), 1000 generations each; the line joins the means. "
         "Base controls are single runs.", ha="center", fontsize=9, color="#666")
out="artifacts/scale_ladder.png"
_png, _pdf = guardar(fig, out, dpi=150)
print("->", out)
