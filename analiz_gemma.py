#!/usr/bin/env python3
# Análisis de bleeding + capacidad + estilo para las corridas gemma (Mg1 AXIS, Mg3 NEXUS).
# Lee termometro.jsonl (capacidad/estilo por gen) y campeones_por_generacion.json (genoma por gen).
import json, re, statistics as st
from estilo import GLIFOS, LEXICONES

RUNS = {
    "AxisGemma (Mg1)":     ("results/MIA/AxisGemma/Mg1_AxisGemma_Banco", "axis",     "modelfiles_gemma/AxisGemma.modelfile"),
    "NexusGemma (Mg3)":    ("results/MIA/NexusGemma/Mg3_NexusGemma_Banco", "nexus",   "modelfiles_gemma/NexusGemma.modelfile"),
    "BioLlamaGemma (Mg4)": ("results/MIA/BioLlamaGemma/Mg4_BioLlamaGemma_Banco", "biollama", "modelfiles_gemma/BioLlamaGemma.modelfile"),
    "AxisQwen (M1) [ref]":     ("results/MIA/AxisQwen/M1_AxisQwen_Banco", "axis",       "axis_qwen_3.1.txt"),
    "NexusQwen (M3) [ref]":    ("results/MIA/NexusQwen/M3_NexusQwen_Banco", "nexus",     "NO_MODELFILE"),
    "BioLlamaQwen (M4) [ref]": ("results/MIA/BioLlamaQwen/M4_BioLlamaQwen_Banco", "biollama", "modelfiles_gemma/BioLlamaGemma.modelfile"),
}

def b64_lines(modelfile):
    try:
        txt = open(modelfile, encoding="utf-8").read()
    except FileNotFoundError:
        return []
    return [m for m in re.findall(r"[A-Za-z0-9+/]{20,}={0,2}", txt)]

def geno_texto(g):
    """Serializa un genotipo a texto plano para escanear firmas."""
    partes = [g.get("rol",""), g.get("formato_salida","")]
    partes += g.get("estrategias",[]) or []
    partes += g.get("restricciones",[]) or []
    return " \n ".join(str(p) for p in partes)

def scan(texto, adn):
    tl = texto.lower()
    nglifos = sum(texto.count(x) for x in GLIFOS)
    nb64 = sum(1 for b in adn if b[:24] in texto)  # réplica literal (prefijo de 24 chars)
    lex = {k: sum(tl.count(s) for s in v) for k,v in LEXICONES.items()}
    return nglifos, nb64, lex

def pct(n, d): return 100.0*n/d if d else 0.0

for nombre,(d,idkey,mf) in RUNS.items():
    print("="*70); print(f" {nombre}"); print("="*70)
    adn = b64_lines(mf)

    # ---------- CAPACIDAD + ESTILO (termómetro) ----------
    term = [json.loads(l) for l in open(f"{d}/termometro.jsonl") if l.strip()]
    fh = [t["fitness_heldout"] for t in term]
    print(f"\n[CAPACIDAD held-out]  n={len(term)} mediciones (cada 25 gen)")
    print(f"  inicio (gen {term[0]['generacion']}): {fh[0]:.3f}   fin (gen {term[-1]['generacion']}): {fh[-1]:.3f}")
    print(f"  min: {min(fh):.3f}   max: {max(fh):.3f}   media: {st.mean(fh):.3f}")
    delta = fh[-1]-fh[0]
    print(f"  Δ(fin-inicio): {delta:+.3f}  -> {'DEGRADA' if delta<-0.03 else 'MEJORA' if delta>0.03 else 'ESTABLE'}")
    cat = term[-1]["por_categoria"]
    print("  por categoría (final):", ", ".join(f"{k}={v:.2f}" for k,v in cat.items()))

    est_i, est_f = term[0]["estilo"], term[-1]["estilo"]
    print(f"\n[ESTILO en RESPUESTAS - termómetro] (inicio -> fin)")
    for k in ["glifos","lex_axis","lex_nexus","lex_biollama","longitud"]:
        print(f"  {k:14s}: {est_i.get(k,0):.3f} -> {est_f.get(k,0):.3f}")

    # ---------- BLEEDING EN EL GENOMA (campeones por generación) ----------
    camp = json.load(open(f"{d}/campeones_por_generacion.json"))
    ng=nb=0; on_g=on_b=None; glifo_gens=[]; b64_gens=[]
    lex_tot={"axis":0,"nexus":0,"biollama":0}
    for c in camp:
        txt = geno_texto(c["genotipo"])
        g,b,lx = scan(txt, adn)
        if g>0:
            ng+=1; glifo_gens.append(c["generacion"])
            if on_g is None: on_g=c["generacion"]
        if b>0:
            nb+=1; b64_gens.append(c["generacion"])
            if on_b is None: on_b=c["generacion"]
        for k in lex_tot: lex_tot[k]+=lx[k]
    N=len(camp)
    print(f"\n[BLEEDING EN EL GENOMA]  {N} campeones (1 por gen)")
    print(f"  glifos:  {ng}/{N} campeones = {pct(ng,N):.1f}%   onset gen {on_g}   (rango {min(glifo_gens) if glifo_gens else '-'}-{max(glifo_gens) if glifo_gens else '-'})")
    print(f"  base64 ADN literal: {nb}/{N} = {pct(nb,N):.1f}%   onset gen {on_b}")
    dom = max(lex_tot, key=lex_tot.get)
    print(f"  léxico acumulado en genoma: axis={lex_tot['axis']} nexus={lex_tot['nexus']} biollama={lex_tot['biollama']}  -> dominante: {dom.upper()}")

    # último tercio (¿persiste tras converger?)
    ult = camp[-333:]
    ng2=sum(1 for c in ult if scan(geno_texto(c['genotipo']),adn)[0]>0)
    print(f"  glifos en último tercio (gen ~667-1000): {pct(ng2,len(ult)):.1f}%")
    print()
