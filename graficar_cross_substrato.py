#!/usr/bin/env python3
# Figura cross-substrato (3 identidades × 2 substratos): el CANAL del identity bleeding
# depende del substrato. (A) bleeding simbólico en el genoma: qwen coloniza (AXIS/BIO),
# gemma queda limpio (las 3). (B) capacidad held-out: qwen abajo/volátil, gemma alto/estable.
# Paleta Okabe-Ito (CVD-safe). Color = identidad; estilo de línea = substrato.
import json, re
import matplotlib
matplotlib.use("Agg")
from guardar_figura import guardar   # PNG + PDF vectorial (fonttype 42)
import matplotlib.pyplot as plt
from estilo import GLIFOS

COL = {"AXIS": "#D55E00", "NEXUS": "#0072B2", "BIO": "#009E73"}
RUNS = {  # etiqueta: (dir, modelfile_ADN, identidad, substrato, linestyle)
    "AXIS·Qwen":      ("results/MIA/AxisQwen/M1_AxisQwen_Banco",         "axis_qwen_3.1.txt",                     "AXIS",     "Qwen-1.7B",  "-"),
    "AXIS·Gemma":     ("results/MIA/AxisGemma/Mg1_AxisGemma_Banco",      "modelfiles_gemma/AxisGemma.modelfile",  "AXIS",     "Gemma-12B",  "--"),
    "NEXUS·Qwen":     ("results/MIA/NexusQwen/M3_NexusQwen_Banco",       "NO",                                    "NEXUS",    "Qwen-1.7B",  "-"),
    "NEXUS·Gemma":    ("results/MIA/NexusGemma/Mg3_NexusGemma_Banco",    "modelfiles_gemma/NexusGemma.modelfile", "NEXUS",    "Gemma-12B",  "--"),
    "BIO·Qwen":  ("results/MIA/BioLlamaQwen/M4_BioLlamaQwen_Banco", "modelfiles_gemma/BioLlamaGemma.modelfile","BIO","Qwen-1.7B",  "-"),
    "BIO·Gemma": ("results/MIA/BioLlamaGemma/Mg4_BioLlamaGemma_Banco","modelfiles_gemma/BioLlamaGemma.modelfile","BIO","Gemma-12B","--"),
}

def b64(mf):
    try: return re.findall(r"[A-Za-z0-9+/]{20,}={0,2}", open(mf, encoding="utf-8").read())
    except OSError: return []

def geno_txt(g):
    return " ".join(str(x) for x in [g.get("rol",""), g.get("formato_salida","")]
                     + (g.get("estrategias",[]) or []) + (g.get("restricciones",[]) or []))

import os, statistics as st

def medir_dir(d, adn):
    camp = json.load(open(f"{d}/campeones_por_generacion.json")); N = len(camp)
    ng = sum(1 for c in camp if any(x in geno_txt(c["genotipo"]) for x in GLIFOS))
    nb = sum(1 for c in camp if any(b[:24] in geno_txt(c["genotipo"]) for b in adn))
    return 100*ng/N, 100*nb/N

data = {}
for lbl,(d,mf,ident,sub,ls) in RUNS.items():
    adn = b64(mf)
    # el bloque qwen está replicado en 3 seeds (100 = dir base, 101/102 = sufijo _sNNN)
    dirs = [d] + [f"{d}_s{k}" for k in (101,102) if os.path.isdir(f"{d}_s{k}")]
    med = [medir_dir(x, adn) for x in dirs]
    gl_s = [m[0] for m in med]; b6_s = [m[1] for m in med]
    term = [json.loads(l) for l in open(f"{d}/termometro.jsonl") if l.strip()]
    data[lbl] = dict(ident=ident, sub=sub, ls=ls, col=COL[ident],
                     glifo_pct=st.mean(gl_s), b64_pct=st.mean(b6_s),
                     glifo_seeds=gl_s, b64_seeds=b6_s, n_seeds=len(dirs),
                     gens=[t["generacion"] for t in term], cap=[t["fitness_heldout"] for t in term])

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 11})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.4))

# ---- Panel A: bleeding simbólico en el genoma (glifos + base64), pares qwen/gemma por identidad ----
labels = list(RUNS.keys()); x = range(len(labels)); w = 0.4
gl = [data[l]["glifo_pct"] for l in labels]; b6 = [data[l]["b64_pct"] for l in labels]
b1 = axA.bar([i-w/2 for i in x], gl, w, label="Glyphs", color="#CC79A7", edgecolor="black", lw=.6)
b2 = axA.bar([i+w/2 for i in x], b6, w, label="Verbatim base64 persona strings", color="#E69F00", edgecolor="black", lw=.6)
for k, b in enumerate(list(b1)+list(b2)):
    h = b.get_height()
    l = labels[k % len(labels)]
    seeds = data[l]["glifo_seeds" if k < len(labels) else "b64_seeds"]
    tope = max([h] + list(seeds))
    axA.text(b.get_x()+b.get_width()/2, tope+2.5, f"{h:.0f}", ha="center", va="bottom",
             fontsize=8.5, fontweight="bold" if h > 0 else "normal", color="#222")
# puntos por seed encima de la barra: la dispersión entre réplicas ES parte del hallazgo
for i, l in enumerate(labels):
    for off, clave in ((-w/2, "glifo_seeds"), (w/2, "b64_seeds")):
        vals = data[l][clave]
        if len(vals) < 2:      # n=1 (gemma): no hay dispersión que mostrar
            continue
        jit = [-0.09, 0.0, 0.09][:len(vals)]
        axA.plot([i+off+j for j in jit], vals, marker="o", markersize=5.5, linestyle="none",
                 markerfacecolor="white", markeredgecolor="#333", markeredgewidth=1.2, zorder=4)
axA.set_ylabel("% of champion genomes carrying the signature")
axA.set_title("A · SYMBOLIC imprinting in the genome", fontweight="bold", loc="left", pad=18)
axA.annotate("Qwen: bar = mean of 3 seeds, points = each seed · Gemma: single run",
             xy=(0.0, 1.012), xycoords="axes fraction", fontsize=8.5, color="#666")
axA.set_xticks(list(x)); axA.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
axA.set_ylim(0, 88); axA.legend(frameon=False, loc="upper right")
axA.plot([], [], marker="o", markersize=5.5, linestyle="none", markerfacecolor="white",
         markeredgecolor="#333", markeredgewidth=1.2, label="individual seed (qwen, n=3)")
axA.legend(frameon=False, loc="upper right", fontsize=8.5)
axA.annotate("only the SMALL model's genome\nis colonized\n(AXIS and BIO)", xy=(0.2, 74),
             xytext=(1.6, 62), fontsize=8.5, color="#555",
             arrowprops=dict(arrowstyle="->", color="#999"))
axA.set_ylabel("% of champion genomes carrying the signature\n(bar = mean across seeds)")

# ---- Panel B: capacidad held-out en el tiempo ----
for lbl,dd in data.items():
    axB.plot(dd["gens"], dd["cap"], color=dd["col"], ls=dd["ls"], lw=2.0, label=lbl, alpha=.9)
axB.set_xlabel("Generation"); axB.set_ylabel("Held-out capability (fitness)")
axB.set_title("B · Capability: Gemma high and stable, Qwen lower and volatile", fontweight="bold", loc="left")
axB.annotate("seed-100 trajectories", xy=(0.985, 0.02), xycoords="axes fraction",
             fontsize=8.5, color="#777", ha="right")
axB.set_ylim(0.5, 1.0); axB.legend(frameon=False, loc="lower left", ncol=3, fontsize=8)
axB.annotate("Gemma-12B  (≈0.90–0.96)", xy=(600, 0.965), fontsize=9, color="#444")
axB.annotate("Qwen-1.7B", xy=(600, 0.66), fontsize=9, color="#444")

fig.suptitle("The CHANNEL of persona imprinting is substrate-dependent  ·  3 personas × 2 substrates",
             fontsize=13.5, fontweight="bold", y=1.005)
fig.tight_layout()
out = "artifacts/cross_substrate_imprinting.png"
_png, _pdf = guardar(fig, out, dpi=150)
print("guardado:", out, "\nRESUMEN:")
for l in labels:
    print(f"  {l:16s} glifos={data[l]['glifo_pct']:5.1f}%  base64={data[l]['b64_pct']:4.1f}%  "
          f"cap {data[l]['cap'][0]:.2f}->{data[l]['cap'][-1]:.2f} (media {sum(data[l]['cap'])/len(data[l]['cap']):.2f})")
