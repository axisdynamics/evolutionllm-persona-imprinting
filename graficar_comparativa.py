#!/usr/bin/env python3
"""
Figura comparativa del bloque Qwen (C1/M1/M3/M4) para el paper.
Paleta Okabe-Ito (CVD-safe, validada); etiquetas directas (relief de contraste);
un eje por panel; leyenda por identidad. Sin GPU.

Uso: evolution/bin/python graficar_comparativa.py
"""
import json, re, base64
from pathlib import Path
from collections import defaultdict
import statistics as st
import matplotlib
matplotlib.use("Agg")
from guardar_figura import guardar   # PNG + PDF vectorial (fonttype 42)
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
RUNS = {
    "C1 (base)":     "results/Modelo Base/Qwen3 1.7B/C1_Qwen3_Banco",
    "M1 (AXIS)":     "results/MIA/AxisQwen/M1_AxisQwen_Banco",
    "M3 (NEXUS)":    "results/MIA/NexusQwen/M3_NexusQwen_Banco",
    "M4 (BIO)": "results/MIA/BioLlamaQwen/M4_BioLlamaQwen_Banco",
}
# Okabe-Ito: gris neutro para el control, hues distintos por identidad (CVD ΔE>>12)
COL = {"C1 (base)": "#6E6E6E", "M1 (AXIS)": "#D55E00", "M3 (NEXUS)": "#0072B2", "M4 (BIO)": "#009E73"}
GLIFOS = set("○→∞◊∇☆◍◎═●◆▲▼♦∴∵≡⊕⊗⊙∆ΨΩΘΗΣΔΦΓΛ")
B64 = re.compile(r'[A-Za-z0-9+/]{24,}={0,2}')

SEEDS = [100, 101, 102]
def _dir(base, seed): return REPO / (base if seed == 100 else f"{base}_s{seed}")
# {run: {seed: filas del termómetro}} — el bloque qwen está replicado en 3 seeds
T = {k: {s: [json.loads(l) for l in open(_dir(v, s) / "termometro.jsonl")]
         for s in SEEDS if (_dir(v, s) / "termometro.jsonl").exists()}
     for k, v in RUNS.items()}
def finmed(rows, f): return st.mean([f(r) for r in rows[-10:]])
def por_seed(run, f):   """valor final (media de las últimas 10 mediciones) en cada seed"""; return [finmed(rows, f) for rows in T[run].values()]

# bleeding % desde campeones
def bleeding_pct(run_dir):
    camps = json.load(open(REPO / run_dir / "campeones_por_generacion.json"))
    g = b = 0
    for c in camps:
        t = c["genotipo"].get("rol","") + " ".join(c["genotipo"].get("estrategias",[]))
        if any(x in t for x in GLIFOS): g += 1
        for m in B64.findall(t):
            try:
                d = base64.b64decode(m+"="*(-len(m)%4)).decode("utf-8")
                if len(d) >= 8 and any(ch.isalpha() for ch in d): b += 1; break
            except Exception: pass
    n = len(camps)
    return 100*g/n, 100*b/n

# bleeding: se toma de analiz_multiseed (MISMO detector que las figuras 3 y 4; tener dos
# contadores de glifos distintos hacía que cada figura reportara un número diferente).
from analiz_multiseed import recolectar as _recolectar
_MS = _recolectar()
_MAP = {"C1 (base)": "C1 Qwen3 base", "M1 (AXIS)": "M1 AxisQwen",
        "M3 (NEXUS)": "M3 NexusQwen", "M4 (BIO)": "M4 BioLlamaQwen"}
BL_S = {k: [(m["glifos"], m["b64"]) for m in _MS[_MAP[k]].values()] for k in RUNS}
BL = {k: (st.mean([x[0] for x in v]), st.mean([x[1] for x in v])) for k, v in BL_S.items()}
runs = list(RUNS)

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(2, 2, figsize=(14, 9))

def barras_agrupadas(ax, categorias, valores_por_run, titulo, ylabel, ymax=None, fmt="%.2f",
                     puntos_por_run=None):
    x = range(len(categorias)); w = 0.2
    for i, r in enumerate(runs):
        off = (i - 1.5) * w
        bars = ax.bar([xi + off for xi in x], valores_por_run[r], w, label=r, color=COL[r],
                      edgecolor="white", linewidth=1.2)
        ax.bar_label(bars, fmt=fmt, fontsize=6.5, padding=1)
        if puntos_por_run:                       # una marca por seed encima de su barra
            for xi, vals in zip(x, puntos_por_run[r]):
                jit = [-0.045, 0.0, 0.045][:len(vals)]
                ax.plot([xi + off + j for j in jit], vals, marker="o", markersize=3.4,
                        linestyle="none", markerfacecolor="white", markeredgecolor="#333",
                        markeredgewidth=0.9, zorder=5)
    ax.set_xticks(list(x)); ax.set_xticklabels([{"extraccion":"extraction","formato":"format","factual":"factual",
                         "math":"math"}.get(c, c) for c in categorias], fontsize=9)
    ax.set_title(titulo, fontsize=11); ax.set_ylabel(ylabel, fontsize=9)
    if ymax: ax.set_ylim(0, ymax)

# A: capacidad held-out (identidad iguala/supera al base)
fh = lambda x: x["fitness_heldout"]
axA = axs[0,0]
for i, r in enumerate(runs):
    vals = por_seed(r, fh)
    for j, v in enumerate(vals):
        axA.plot(i + (j - 1) * 0.13, v, marker="o", markersize=9, color=COL[r],
                 linestyle="none", zorder=3)
    m = st.mean(vals)
    axA.plot([i - 0.28, i + 0.28], [m, m], color=COL[r], lw=2.6, alpha=0.55, zorder=2)
    axA.annotate(f"{m:.3f}", (i + 0.32, m), fontsize=9, color="#333", va="center")
axA.set_xticks(range(len(runs))); axA.set_xticklabels(runs, fontsize=9)
axA.set_title("Capability — indistinguishable across conditions", fontsize=11)
axA.set_ylabel("held-out fitness (final)", fontsize=9); axA.set_xlim(-0.5, len(runs) - 0.2)

# B: firma de estilo (cada identidad distinta)
rasgos = ["glyphs", "lex_axis", "lex_biollama", "length/20"]
def styleval_seeds(r, rasgo):
    if rasgo == "length/20": return [v/20 for v in por_seed(r, lambda x: x["estilo"]["longitud"])]
    return por_seed(r, lambda x: x["estilo"]["glifos" if rasgo == "glyphs" else rasgo])
stB_s = {r: [styleval_seeds(r, g) for g in rasgos] for r in runs}
stB = {r: [st.mean(v) for v in stB_s[r]] for r in runs}
barras_agrupadas(axs[0,1], rasgos, stB, "Style fingerprint — one per persona", "value", None,
                 "%.2f", puntos_por_run=stB_s)
axs[0,1].annotate("lex_biollama is not a clean signal (the base scores high) — see §4.1",
                  xy=(0.0, -0.20), xycoords="axes fraction", fontsize=7, color="#777")
axs[0,1].legend(fontsize=7, ncol=2)

# C: efecto de identidad en capacidad (robusto)
catsC = ["extraccion", "formato", "math", "factual"]
capC_s = {r: [por_seed(r, lambda x, c=c: x["por_categoria"].get(c, 0)) for c in catsC] for r in runs}
capC = {r: [st.mean(v) for v in capC_s[r]] for r in runs}
barras_agrupadas(axs[1,0], catsC, capC, "Persona effect: \u2193extraction, \u2191format/math/factual",
                 "capability", 1.0, "%.2f", puntos_por_run=capC_s)
axs[1,0].legend(fontsize=7, ncol=2)

# D: canal de bleeding (% de generaciones)
canales = ["% champions with glyphs", "% champions with verbatim base64"]
blD = {r: [BL[r][0], BL[r][1]] for r in runs}
blD_s = {r: [[x[0] for x in BL_S[r]], [x[1] for x in BL_S[r]]] for r in runs}
barras_agrupadas(axs[1,1], canales, blD, "Imprinting channel: always present, magnitude varies by seed",
                 "% of champion genomes", 88, "%.0f", puntos_por_run=blD_s)
axs[1,1].legend(fontsize=7, ncol=2)

fig.suptitle("Qwen block (3 seeds): equal capability, distinct persona signature",
             fontsize=13, y=1.0)
fig.text(0.5, -0.01, "Bars are the mean of seeds 100/101/102; each white marker is one seed.",
         ha="center", fontsize=9, color="#666")
fig.tight_layout()
out = REPO / "artifacts" / "qwen_block_overview.png"
_png, _pdf = guardar(fig, out, dpi=140)
print(f"💾 {out}")
