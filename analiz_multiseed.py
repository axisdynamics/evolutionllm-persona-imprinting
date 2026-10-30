#!/usr/bin/env python3
# Multi-seed del bloque qwen (seeds 100/101/102): ¿replica el identity bleeding?
# Mismo detector que analiz_gemma.py (glifos + ADN base64 literal, prefijo de 24 chars).
# Uso: evolution/bin/python analiz_multiseed.py
import json, re, statistics as st
from pathlib import Path
from estilo import GLIFOS, LEXICONES

# cond -> (dir_base, modelfile con el ADN)
COND = {
    "M1 AxisQwen":      ("results/MIA/AxisQwen/M1_AxisQwen_Banco",           "axis_qwen_3.1.txt"),
    "M3 NexusQwen":     ("results/MIA/NexusQwen/M3_NexusQwen_Banco",         "nexusvex_deep_q3.txt"),
    "M4 BioLlamaQwen":  ("results/MIA/BioLlamaQwen/M4_BioLlamaQwen_Banco",   "biovex_deep_q3.txt"),
    "A3 AxisQwen noIM": ("results/MIA/AxisQwen/A3_AxisQwen_noIM_Banco",      "axis_qwen_3.1.txt"),
    "C1 Qwen3 base":    ("results/Modelo Base/Qwen3 1.7B/C1_Qwen3_Banco",    None),
}
SEEDS = [100, 101, 102]   # 100 = corrida original (sufijo vacío)

def b64_lines(mf):
    if not mf: return []
    try: return re.findall(r"[A-Za-z0-9+/]{20,}={0,2}", open(mf, encoding="utf-8").read())
    except FileNotFoundError: return []

def geno_texto(g):
    p = [g.get("rol",""), g.get("formato_salida","")]
    p += g.get("estrategias",[]) or []
    p += g.get("restricciones",[]) or []
    return " \n ".join(str(x) for x in p)

def medir(d, adn):
    camp = json.load(open(f"{d}/campeones_por_generacion.json"))
    ng = nb = 0; on_g = on_b = None
    for c in camp:
        txt = geno_texto(c["genotipo"])
        if sum(txt.count(x) for x in GLIFOS) > 0:
            ng += 1; on_g = on_g if on_g is not None else c["generacion"]
        if sum(1 for b in adn if b[:24] in txt) > 0:
            nb += 1; on_b = on_b if on_b is not None else c["generacion"]
    N = len(camp)
    term = [json.loads(l) for l in open(f"{d}/termometro.jsonl") if l.strip()]
    fh = [t["fitness_heldout"] for t in term]
    return dict(N=N, glifos=100*ng/N, b64=100*nb/N, on_g=on_g, on_b=on_b,
                cap_med=st.mean(fh), cap_ini=fh[0], cap_fin=fh[-1], n_term=len(fh))

def recolectar():
    """{condición: {seed: medida}} — fuente única para la tabla y las figuras."""
    out = {}
    for cond, (base, mf) in COND.items():
        adn = b64_lines(mf); out[cond] = {}
        for s in SEEDS:
            d = base if s == 100 else f"{base}_s{s}"
            if Path(f"{d}/campeones_por_generacion.json").exists():
                out[cond][s] = medir(d, adn)
    return out


def main():
    print(f"{'condición':<19}{'seed':>5}{'glifos%':>9}{'base64%':>9}{'onset':>7}"
          f"{'cap media':>11}{'cap ini→fin':>15}")
    print("-"*76)
    resumen = {}
    for cond, (base, mf) in COND.items():
        adn = b64_lines(mf); filas = []
        for s in SEEDS:
            d = base if s == 100 else f"{base}_s{s}"
            if not Path(f"{d}/campeones_por_generacion.json").exists():
                print(f"{cond:<19}{s:>5}   (falta)"); continue
            r = medir(d, adn); filas.append(r)
            print(f"{cond:<19}{s:>5}{r['glifos']:>9.1f}{r['b64']:>9.1f}"
                  f"{str(r['on_g'] or '-'):>7}{r['cap_med']:>11.3f}"
                  f"{r['cap_ini']:>7.3f}→{r['cap_fin']:.3f}")
        if filas:
            resumen[cond] = filas
            g = [f['glifos'] for f in filas]; b = [f['b64'] for f in filas]; c = [f['cap_med'] for f in filas]
            sd = lambda v: st.stdev(v) if len(v) > 1 else 0.0
            print(f"{'  → media ± sd':<19}{'':>5}{st.mean(g):>6.1f}±{sd(g):<2.0f}{st.mean(b):>6.1f}±{sd(b):<2.0f}"
                  f"{'':>7}{st.mean(c):>8.3f}±{sd(c):.3f}")
            print("-"*76)


if __name__ == "__main__":
    main()
