# estilo.py
# Medición de ESTILO COGNITIVO independiente de la correctitud.
#
# Motivación: los avatares MIA (AXIS contemplativo, NEXUS sistemático, BioLlama
# biológico) se conciben como ESTILOS cognitivos distintos, no solo niveles de
# capacidad. Un validador de "acertó/no acertó" no captura eso. Aquí extraemos,
# del TEXTO de respuesta, un vector de rasgos estilísticos que forman la "huella"
# cognitiva de un genoma. Se calcula sobre las mismas generaciones que ya se hacen
# para evaluar fitness → costo GPU cero adicional.
#
# Rasgos (normalizados a "por 100 palabras" donde tiene sentido):
#   - Sistematicidad (NEXUS): enumeracion, marcadores_seq, descomposicion, conectores_logicos
#   - Expansión (AXIS):       longitud, dens_frase, parrafos
#   - Firma léxica:           lex_axis, lex_nexus, lex_biollama
#   - Simbólico (AXIS):       glifos
#   - Figuración/abstracción: metafora, abstraccion
#   - (Anti-concisión):       preambulo

import re

GLIFOS = set("○→∞◊∇☆◍◎═●◆▲▼♦∴∵≡⊕⊗⊙")  # firma simbólica esotérica

LEXICONES = {
    "axis": ["concienc", "conscien", "onda", "cuánt", "cuant", "vibrac", "resonanc",
             "infinit", "cósmic", "cosmic", "silenci", "alma", "puente", "frecuenc",
             "holográf", "holograf", "espíritu", "trascend", "fluir", "etern", "sagrad"],
    "nexus": ["sistem", "red ", "nodo", "estructur", "patrón", "patron", "descompon",
              "component", "conexión", "conexion", "jerarq", "proceso", "módulo",
              "modul", "iterati", "algoritm", "optimiz", "analiz", "metódic", "metodic"],
    "biollama": ["evoluc", "organism", "célul", "celul", "genétic", "genetic", "adaptac",
                 "adapta", "biológ", "biolog", "mutac", "especie", "ecosistem", "orgánic",
                 "organic", "adn", "metaboli", "simbiosis", "natural"],
}

_SEQ = [r'\bpaso\s*\d+', r'\bprimero\b', r'\bsegundo\b', r'\btercero\b', r'\bluego\b',
        r'\bentonces\b', r'\bdespués\b', r'\bfinalmente\b', r'\ba continuación\b']
_CONECT = [r'\bpor lo tanto\b', r'\bporque\b', r'\bdado que\b', r'\bpuesto que\b',
           r'\ben consecuencia\b', r'\bpor ende\b', r'\bya que\b', r'\bpor consiguiente\b']
_PREAMB = [r'\bcreo que\b', r'\ben mi opinión\b', r'\bes importante\b', r'\bdéjame\b',
           r'\binteresante\b', r'\bcabe destacar\b', r'\bpermíteme\b', r'\bme gustaría\b']
_META = [r'\bcomo si\b', r'\bcual si\b', r'\bes decir\b', r'\bse asemeja\b',
         r'\bmetáfora\b', r'\bimagina\b', r'\bcomo un\b', r'\bcomo una\b']
_DESC = ["descompon", "dividir", "subproblema", "sub-problema", "por partes",
         "componente", "por separado", "paso a paso", "desglos"]


def _words(t):
    return re.findall(r'\w+', t.lower())


def extraer_features(texto: str) -> dict:
    t = (texto or "").strip()
    tl = t.lower()
    words = _words(t)
    nw = max(len(words), 1)
    sents = [s for s in re.split(r'[.!?]+', t) if s.strip()]

    enum = len(re.findall(r'(?m)^\s*(?:\d+[.)]|[-•*])\s+', t))
    seq = sum(len(re.findall(p, tl)) for p in _SEQ)
    desc = sum(tl.count(s) for s in _DESC)
    parrafos = len([p for p in t.split("\n\n") if p.strip()])
    conect = 100.0 * sum(len(re.findall(p, tl)) for p in _CONECT) / nw
    preamb = float(sum(len(re.findall(p, tl)) for p in _PREAMB))
    meta = 100.0 * sum(len(re.findall(p, tl)) for p in _META) / nw
    abstrac = 100.0 * sum(1 for w in words if re.search(r'(ción|cion|idad|ismo|ncia|miento)$', w)) / nw
    glifos = sum(t.count(g) for g in GLIFOS)

    f = {
        "longitud": float(nw),
        "dens_frase": nw / max(len(sents), 1),
        "enumeracion": float(enum),
        "marcadores_seq": float(seq),
        "descomposicion": float(desc),
        "parrafos": float(parrafos),
        "conectores_logicos": conect,
        "metafora": meta,
        "abstraccion": abstrac,
        "preambulo": preamb,
        "glifos": float(glifos),
    }
    for k, lex in LEXICONES.items():
        f[f"lex_{k}"] = 100.0 * sum(tl.count(s) for s in lex) / nw
    return f


def promediar(lista: list) -> dict:
    if not lista:
        return {}
    keys = lista[0].keys()
    return {k: sum(d[k] for d in lista) / len(lista) for k in keys}


# Sub-índices interpretables (combinaciones de rasgos) para reportar "estilo"
def indices_estilo(feat: dict) -> dict:
    """Resume el vector en 3 ejes interpretables de estilo cognitivo."""
    return {
        "sistematicidad": feat.get("enumeracion", 0) + feat.get("marcadores_seq", 0)
                          + feat.get("descomposicion", 0) + 0.1 * feat.get("conectores_logicos", 0),
        "expansion": feat.get("longitud", 0) / 100.0 + feat.get("parrafos", 0),
        "esoterismo_axis": feat.get("glifos", 0) + feat.get("lex_axis", 0),
    }
