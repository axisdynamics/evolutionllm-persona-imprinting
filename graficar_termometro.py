#!/usr/bin/env python3
"""
Figuras del termómetro de un run: capacidad, escalera de dificultad, estilo, categorías.
Genérico y reusable (C1, M1, M3, ...). No requiere GPU.

Uso: evolution/bin/python graficar_termometro.py "<run_dir>"
"""
import json, sys, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = Path(sys.argv[1])
T = [json.loads(l) for l in open(run / "termometro.jsonl")]
gen = [p["generacion"] for p in T]
nombre = run.name

fig, axs = plt.subplots(2, 2, figsize=(14, 9))

# A: fitness held-out (capacidad comparable)
axs[0, 0].plot(gen, [p["fitness_heldout"] for p in T], "-o", ms=3, color="tab:blue")
axs[0, 0].set_title(f"{nombre}: fitness held-out (capacidad comparable)")
axs[0, 0].set_xlabel("generación"); axs[0, 0].set_ylabel("fitness")
axs[0, 0].set_ylim(0, 1); axs[0, 0].grid(alpha=0.3)

# B: escalera de dificultad calibrada
for b in range(1, 8):
    ys = [p["por_dificultad_bin"].get(str(b)) for p in T]
    axs[0, 1].plot(gen, [y if y is not None else float("nan") for y in ys], label=f"bin{b}")
axs[0, 1].set_title("Capacidad por bin de dificultad calibrada (1=fácil, 7=difícil)")
axs[0, 1].set_xlabel("generación"); axs[0, 1].set_ylim(0, 1)
axs[0, 1].legend(fontsize=7, ncol=2); axs[0, 1].grid(alpha=0.3)

# C: estilo — ¿emerge identidad?
for r, col in [("glifos", "tab:red"), ("lex_axis", "tab:orange"),
               ("lex_nexus", "tab:green"), ("enumeracion", "tab:purple")]:
    axs[1, 0].plot(gen, [p["estilo"][r] for p in T], label=r, color=col)
ax2 = axs[1, 0].twinx()
ax2.plot(gen, [p["estilo"]["longitud"] for p in T], ":", color="tab:blue", alpha=0.5, label="longitud")
ax2.set_ylabel("longitud (palabras)", color="tab:blue")
axs[1, 0].set_title("Estilo cognitivo (¿emerge identidad?)")
axs[1, 0].set_xlabel("generación"); axs[1, 0].legend(fontsize=7, loc="upper left"); axs[1, 0].grid(alpha=0.3)

# D: capacidad por categoría, inicio vs final
cats = sorted(T[0]["por_categoria"])
ini = [st.mean([p["por_categoria"][c] for p in T[:10] if c in p["por_categoria"]]) for c in cats]
fin = [st.mean([p["por_categoria"][c] for p in T[-10:] if c in p["por_categoria"]]) for c in cats]
x = range(len(cats)); w = 0.38
axs[1, 1].bar([i - w / 2 for i in x], ini, w, label="inicio")
axs[1, 1].bar([i + w / 2 for i in x], fin, w, label="final")
axs[1, 1].set_xticks(list(x)); axs[1, 1].set_xticklabels(cats, rotation=40, ha="right", fontsize=8)
axs[1, 1].set_title("Capacidad por categoría: inicio vs final")
axs[1, 1].set_ylim(0, 1); axs[1, 1].legend(); axs[1, 1].grid(alpha=0.3, axis="y")

fig.tight_layout()
out = run / "figuras_termometro.png"
fig.savefig(out, dpi=130)
print(f"💾 {out}")
