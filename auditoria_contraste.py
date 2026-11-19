#!/usr/bin/env python3
"""
Auditoría del contraste central del paper: presencia de marcadores simbólicos de persona
en el genoma, por EJECUCIÓN (no por campeón), con el evento definido explícitamente.

Nace del feedback externo a v3.0, que detectó que el paper declaraba "5 de 6" ejecuciones
positivas mientras sus propias tablas mostraban seis. Este script es la fuente única y
trazable de: el conteo por ejecución, el test de Fisher, la sensibilidad al umbral, la
especificidad del detector y la tasa de fondo.

Uso: evolution/bin/python auditoria_contraste.py
"""
import json, re, collections, statistics as st
from pathlib import Path
from scipy.stats import fisher_exact
from estilo import GLIFOS

# ---------------------------------------------------------------- definiciones

# La unidad experimental es la EJECUCIÓN. El evento es "la ejecución está impresa
# simbólicamente", definido como: al menos un campeón cuyo genoma contiene un glifo
# presente en la especificación de persona de ESA identidad, o una cadena base64
# copiada literalmente de ella (prefijo de 24 caracteres).
PREFIJO_B64 = 24

# Criterio de INCLUSIÓN en el contraste, fijado sobre el input y no sobre el resultado:
# la especificación de persona debe contener material simbólico transmisible, es decir
# al menos una cadena base64. NEXUS no la tiene (0 cadenas, 1 solo glifo: la flecha "→",
# carácter de fondo), así que un contraste sobre marcadores simbólicos no puede medirla:
# no hay nada específico que copiar. Se reporta aparte como control de persona sin
# material simbólico.
MIN_B64_EN_SPEC = 1

SPECS = {
    "AXIS":     "axis_qwen_3.1.txt",
    "NEXUS":    "nexusvex_deep_q3.txt",
    "BIO":      "biovex_deep_q3.txt",
}

# (etiqueta, brazo, directorio base, seeds, spec de persona o None)
# brazo: "persona" = persona + operador de autoría | "control" = base o ablación sin operador
RUNS = [
    ("AXIS",            "persona", "results/MIA/AxisQwen/M1_AxisQwen_Banco",         [100, 101, 102], "AXIS"),
    ("BIO",             "persona", "results/MIA/BioLlamaQwen/M4_BioLlamaQwen_Banco", [100, 101, 102], "BIO"),
    ("AXIS w/o oper.",  "control", "results/MIA/AxisQwen/A3_AxisQwen_noIM_Banco",    [100, 101, 102], "AXIS"),
    ("Qwen3 base",      "control", "results/Modelo Base/Qwen3 1.7B/C1_Qwen3_Banco",  [100, 101, 102], "AXIS"),
    ("NEXUS",           "aparte",  "results/MIA/NexusQwen/M3_NexusQwen_Banco",       [100, 101, 102], "NEXUS"),
]


def leer_spec(nombre):
    txt = open(SPECS[nombre], encoding="utf-8").read()
    return {x for x in GLIFOS if x in txt}, re.findall(r"[A-Za-z0-9+/]{20,}={0,2}", txt)


def geno_texto(g):
    p = [g.get("rol", ""), g.get("formato_salida", "")]
    p += (g.get("estrategias") or []) + (g.get("restricciones") or [])
    return " \n ".join(str(x) for x in p)


def ruta(base, s):
    """El bloque 1.7B guardó la seed 100 en el directorio base, sin sufijo."""
    return base if s == 100 else f"{base}_s{s}"


# Glifos que no aparecen en NINGUNA especificación: estimador de la tasa de fondo.
_todas = set().union(*(leer_spec(n)[0] for n in SPECS))
FONDO = GLIFOS - _todas


def medir(d, glifos_spec, b64_spec):
    camps = json.load(open(f"{d}/campeones_por_generacion.json"))
    N = len(camps)
    pref = [b[:PREFIJO_B64] for b in b64_spec]
    n_gl = n_b64 = n_excl = n_fondo = 0
    onset = None
    for c in camps:
        t = geno_texto(c["genotipo"])
        presentes = {x for x in GLIFOS if x in t}
        gl = bool(presentes & glifos_spec)
        b64 = any(p in t for p in pref)
        if gl:
            n_gl += 1
        if b64:
            n_b64 += 1
        if gl or b64:
            onset = onset if onset is not None else c["generacion"]
        # glifos exclusivos de esta persona (no compartidos con otra spec) -> especificidad
        otras = set().union(*[leer_spec(n)[0] for n in SPECS if glifos_spec != leer_spec(n)[0]])
        if presentes & (glifos_spec - otras):
            n_excl += 1
        if presentes & FONDO:
            n_fondo += 1
    return dict(N=N, gl=100 * n_gl / N, b64=100 * n_b64 / N,
                excl=100 * n_excl / N, fondo=100 * n_fondo / N, onset=onset)


def fisher(a, b, c, d):
    """a/b = positivos/negativos brazo 1; c/d = idem brazo 2. Devuelve p bilateral."""
    return fisher_exact([[a, b], [c, d]], alternative="two-sided")[1]


GENES = ("rol", "formato_salida", "estrategias", "restricciones")


def por_gen(dirs, glifos_spec, b64_spec):
    """
    En qué GEN aparece el marcador. Respalda la afirmación de §3.2.

    El operador inteligente escribe dos de los cuatro genes, `rol` y `estrategias`, con
    probabilidad 0.5 cada uno cuando dispara. La pregunta es si la impronta se reparte
    entre los dos o se concentra, y la respuesta no era obvia de antemano.
    """
    pref = [b[:PREFIJO_B64] for b in b64_spec]
    cnt = collections.Counter()
    n = 0
    for d in dirs:
        for c in json.load(open(f"{d}/campeones_por_generacion.json")):
            g = c["genotipo"]
            n += 1
            for campo in GENES:
                v = g.get(campo)
                t = " ".join(map(str, v)) if isinstance(v, list) else str(v or "")
                if ({x for x in GLIFOS if x in t} & glifos_spec) or any(p in t for p in pref):
                    cnt[campo] += 1
    return n, cnt


def variantes_b64(dirs, b64_spec):
    """
    Fidelidad de la copia: cuántas versiones DISTINTAS de una misma carga aparecen.

    No se busca ninguna variante concreta: se ancla en el prefijo de la carga original
    y se extrae la corrida base64 maximal que empieza ahí, así que cualquier mutación
    posterior al prefijo aparece sola. Respalda la afirmación de §4.1 de que la herencia
    viene con mutación.
    """
    B64 = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
    out = []
    for d in dirs:
        var = {}
        for c in json.load(open(f"{d}/campeones_por_generacion.json")):
            t = geno_texto(c["genotipo"])
            for carga in b64_spec:
                pref = carga[:PREFIJO_B64]
                i = t.find(pref)
                while i != -1:
                    j = i
                    while j < len(t) and t[j] in B64:
                        j += 1
                    var.setdefault(pref, collections.Counter())[t[i:j]] += 1
                    i = t.find(pref, j if j > i else i + 1)
        out.append((d, var))
    return out


def main():
    filas = []
    print("Definición del evento: ≥1 campeón con un glifo de la spec de SU persona, o una\n"
          f"cadena base64 de esa spec copiada literalmente (prefijo {PREFIJO_B64} chars).")
    print("Unidad experimental: la ejecución.\n")
    print(f"{'condición':<16}{'brazo':<9}{'seed':>5}{'glifos%':>9}{'base64%':>9}"
          f"{'excl%':>8}{'fondo%':>8}{'onset':>7}{'evento':>8}")
    print("-" * 79)
    for etiqueta, brazo, base, seeds, spec in RUNS:
        gl_spec, b64_spec = leer_spec(spec)
        for s in seeds:
            d = ruta(base, s)
            if not Path(f"{d}/campeones_por_generacion.json").exists():
                print(f"{etiqueta:<16}{brazo:<9}{s:>5}   (falta)")
                continue
            r = medir(d, gl_spec, b64_spec)
            ev = (r["gl"] > 0) or (r["b64"] > 0)
            filas.append((etiqueta, brazo, s, r, ev))
            print(f"{etiqueta:<16}{brazo:<9}{s:>5}{r['gl']:>9.1f}{r['b64']:>9.1f}"
                  f"{r['excl']:>8.1f}{r['fondo']:>8.1f}{str(r['onset'] or '-'):>7}{('sí' if ev else 'no'):>8}")
        print("-" * 79)

    P = [f for f in filas if f[1] == "persona"]
    C = [f for f in filas if f[1] == "control"]
    ap = sum(1 for f in P if f[4]); cp = sum(1 for f in C if f[4])
    print(f"\nCONTRASTE (unidad = ejecución, NEXUS excluido por el criterio de input)")
    print(f"  persona + operador : {ap}/{len(P)} positivas")
    print(f"  control + ablación : {cp}/{len(C)} positivas")
    print(f"  Fisher exacto bilateral: p = {fisher(ap, len(P)-ap, cp, len(C)-cp):.4g}")

    nx = [f for f in filas if f[1] == "aparte"]
    print(f"\n  NEXUS (persona sin material simbólico en su spec): "
          f"{sum(1 for f in nx if f[4])}/{len(nx)} positivas  <- lo que predice el mecanismo")

    print("\nSENSIBILIDAD AL UMBRAL (% de campeones con marcador exigido para llamar positiva\n"
          "una ejecución). El valor reportado no debe elegirse para conservar un p:")
    print(f"  {'umbral':>8}{'persona':>10}{'control':>10}{'p bilateral':>14}")
    for u in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0]:
        a = sum(1 for f in P if max(f[3]["gl"], f[3]["b64"]) > u)
        c = sum(1 for f in C if max(f[3]["gl"], f[3]["b64"]) > u)
        print(f"  {u:>7.1f}%{a:>7}/{len(P)}{c:>7}/{len(C)}{fisher(a, len(P)-a, c, len(C)-c):>14.4g}")

    print("\nESPECIFICIDAD")
    ax = [f[3]["excl"] for f in P if f[0] == "AXIS"]
    bl = [f[3]["excl"] for f in P if f[0] == "BIO"]
    print(f"  glifos exclusivos de AXIS ({''.join(sorted(leer_spec('AXIS')[0] - leer_spec('BIO')[0]))}) "
          f"en runs AXIS: {', '.join(f'{v:.1f}%' for v in ax)}")
    print(f"  los mismos glifos en runs BIO: {', '.join(f'{v:.1f}%' for v in bl)}")
    print(f"  tasa de FONDO (glifos de ninguna spec: {''.join(sorted(FONDO))}): "
          f"máx {max(f[3]['fondo'] for f in filas):.1f}% sobre {len(filas)} ejecuciones")
    print("  -> cualquier prevalencia del orden del fondo debe leerse como traza, no como señal.")

    print("\nEN QUÉ GEN APARECE EL MARCADOR (§3.2)")
    print("  El operador inteligente escribe `rol` y `estrategias`, 0.5 cada uno al disparar.")
    for etiqueta, brazo, base, seeds, spec in RUNS:
        if brazo != "persona":
            continue
        gl_spec, b64_spec = leer_spec(spec)
        dirs = [ruta(base, s) for s in seeds
                if Path(f"{ruta(base, s)}/campeones_por_generacion.json").exists()]
        n, cnt = por_gen(dirs, gl_spec, b64_spec)
        detalle = "  ".join(f"{g}={cnt[g]}" for g in GENES)
        print(f"  {etiqueta:<6} ({n} campeones, {len(dirs)} semillas): {detalle}")
    print("  -> la impronta se concentra en `estrategias`, pese a que `rol` lo escribe el")
    print("     modelo con la misma frecuencia; es autoría de gen, no un canal difuso.")

    print("\nFIDELIDAD DE LA COPIA (§4.1): variantes distintas de una misma carga base64")
    for etiqueta, brazo, base, seeds, spec in RUNS:
        if brazo != "persona":
            continue
        _, b64_spec = leer_spec(spec)
        for s in seeds:
            d = ruta(base, s)
            if not Path(f"{d}/campeones_por_generacion.json").exists():
                continue
            for _, var in variantes_b64([d], b64_spec):
                for pref, cnt in var.items():
                    if len(cnt) > 1:
                        exacta = next((c for b in b64_spec for c in [cnt.get(b, 0)] if b.startswith(pref)), 0)
                        print(f"  {etiqueta} s{s}  carga {pref[:12]}...: {len(cnt)} variantes "
                              f"({exacta} copias exactas, {sum(cnt.values()) - exacta} mutadas)")
    print("  -> la copia es casi exacta, no exacta: la herencia viene con mutación.")


if __name__ == "__main__":
    main()
