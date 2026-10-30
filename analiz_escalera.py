#!/usr/bin/env python3
"""
Escalera de escala qwen3 (0.6B / 1.7B / 8B): ¿la colonización simbólica del genoma
desaparece al subir de tamaño DENTRO de la misma familia?

Mismo detector que analiz_multiseed.py (glifos + ADN base64 literal, prefijo 24 chars).
Uso: evolution/bin/python analiz_escalera.py
"""
import json, re, statistics as st
from pathlib import Path
from estilo import GLIFOS, LEXICONES

MF = "axis_qwen_3.1.txt"   # las cadenas base64 son idénticas en los 3 tamaños
ESCALA = [
    ("0.6B", "results/MIA/AxisQwen06b/E06_AxisQwen06b", [100,101,102],
             "results/Modelo Base/Qwen3 0.6B/Cb06_Qwen3_06B_s100"),
    ("1.7B", "results/MIA/AxisQwen/M1_AxisQwen_Banco",  [100,101,102],
             "results/Modelo Base/Qwen3 1.7B/C1_Qwen3_Banco"),
    ("8B",   "results/MIA/AxisQwen8b/E8_AxisQwen8b",    [100,101,102],
             "results/Modelo Base/Qwen3 8B/Cb8_Qwen3_8B_s100"),
]

def adn(): return re.findall(r"[A-Za-z0-9+/]{20,}={0,2}", open(MF, encoding="utf-8").read())
def geno(g):
    p=[g.get("rol",""),g.get("formato_salida","")]+(g.get("estrategias") or [])+(g.get("restricciones") or [])
    return " \n ".join(str(x) for x in p)

def medir(d, A):
    camps = json.load(open(f"{d}/campeones_por_generacion.json"))
    ng=nb=0; on=None; lex=0
    for c in camps:
        t = geno(c["genotipo"])
        if sum(t.count(x) for x in GLIFOS) > 0:
            ng += 1; on = on if on is not None else c["generacion"]
        if sum(1 for b in A if b[:24] in t) > 0: nb += 1
        lex += sum(t.lower().count(s) for s in LEXICONES["axis"])
    term=[json.loads(l) for l in open(f"{d}/termometro.jsonl") if l.strip()]
    N=len(camps)
    return dict(gl=100*ng/N, b64=100*nb/N, on=on, lex=lex,
                cap=st.mean([t["fitness_heldout"] for t in term]))

def ruta(base, s):  # el bloque 1.7B usa el sufijo _sNNN salvo la seed 100 (dir base)
    if "M1_AxisQwen_Banco" in base: return base if s==100 else f"{base}_s{s}"
    return f"{base}_s{s}"

A = adn()
print(f"{'tamaño':<8}{'seed':>6}{'glifos%':>9}{'base64%':>9}{'onset':>7}{'lex_axis':>10}{'capacidad':>11}")
print("-"*60)
resumen={}
for nombre, base, seeds, ctrl in ESCALA:
    filas=[]
    for s in seeds:
        d = ruta(base, s)
        if not Path(f"{d}/campeones_por_generacion.json").exists():
            print(f"{nombre:<8}{s:>6}   (falta)"); continue
        r = medir(d, A); filas.append(r)
        print(f"{nombre:<8}{s:>6}{r['gl']:>9.1f}{r['b64']:>9.1f}{str(r['on'] or '-'):>7}{r['lex']:>10}{r['cap']:>11.3f}")
    if filas:
        sd = lambda v: st.stdev(v) if len(v)>1 else 0.0
        g=[f['gl'] for f in filas]; b=[f['b64'] for f in filas]
        c=[f['cap'] for f in filas]; l=[f['lex'] for f in filas]
        resumen[nombre]=(st.mean(g), sd(g), st.mean(b), st.mean(l), st.mean(c))
        print(f"{'  media':<8}{'':>6}{st.mean(g):>6.1f}±{sd(g):<2.0f}{st.mean(b):>6.1f}±{sd(b):<2.0f}"
              f"{'':>7}{st.mean(l):>10.0f}{st.mean(c):>11.3f}")
    if Path(f"{ctrl}/campeones_por_generacion.json").exists():
        r = medir(ctrl, A)
        print(f"{'  base':<8}{'(ctrl)':>6}{r['gl']:>9.1f}{r['b64']:>9.1f}{str(r['on'] or '-'):>7}{r['lex']:>10}{r['cap']:>11.3f}")
    print("-"*60)

print("\nRESUMEN — ¿monotónico con la escala?")
print(f"{'tamaño':<8}{'glifos% (media±sd)':>22}{'base64%':>10}{'lex_axis':>10}{'capacidad':>11}")
for n,(g,s,b,l,c) in resumen.items():
    print(f"{n:<8}{g:>16.1f} ± {s:<4.0f}{b:>10.1f}{l:>10.0f}{c:>11.3f}")
