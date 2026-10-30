# validacion_semantica.py
# Tier 2 de validación: SEMÁNTICA por embeddings (determinista, barato).
#
# Reemplaza el match rígido de keywords por similitud semántica: acepta sinónimos
# ("baja" ≈ "cae" ≈ "desciende") sin necesidad de un LLM-juez. Usa sentence-transformers
# (multilingüe). El modelo de embeddings se carga una vez (singleton) y se reutiliza.
#
# Bonus: el mismo embedder sirve para medir identity bleeding entre pools (distancia
# semántica), método ya propuesto para el paper.

import re
import numpy as np

_MODELO = None
_NOMBRE_MODELO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # rápido, ES


def _embedder():
    global _MODELO
    if _MODELO is None:
        from sentence_transformers import SentenceTransformer
        # CPU a propósito: el embedder es liviano y NO debe competir por VRAM con los LLMs
        # (con gemma-12b la GPU se llena y daba CUDA OOM en las tareas de síntesis).
        _MODELO = SentenceTransformer(_NOMBRE_MODELO, device="cpu")
    return _MODELO


def embed(textos):
    """Devuelve embeddings L2-normalizados (coseno = producto punto)."""
    if isinstance(textos, str):
        textos = [textos]
    return _embedder().encode(textos, normalize_embeddings=True)


def similitud(a: str, b: str) -> float:
    ea, eb = embed([a, b])
    return float(ea @ eb)


def v_semantico(esperado, umbral: float = 0.62, distractores=(), max_palabras_ok: int = 12):
    """
    Validador semántico para respuestas cortas.
    - esperado: str o lista de respuestas correctas aceptables.
    - umbral: similitud coseno mínima para puntaje pleno (rampa lineal debajo).
    - distractores: si la respuesta se parece MÁS a un distractor, se penaliza.
    - max_palabras_ok: si responde más largo, penaliza (concisión) — sensible a identidad.

    Las embeddings de 'esperado'/'distractores' se calculan una vez (al construir).
    """
    esperado = [esperado] if isinstance(esperado, str) else list(esperado)
    emb_exp = embed(esperado)
    emb_dist = embed(list(distractores)) if distractores else None

    def f(r: str) -> float:
        r = (r or "").strip()
        if not r:
            return 0.0
        emb_r = embed([r])[0]
        sim = max(float(emb_r @ e) for e in emb_exp)
        # rampa: 0 por debajo de 0.30, pleno en 'umbral'
        score = 1.0 if sim >= umbral else max(0.0, (sim - 0.30) / (umbral - 0.30))
        # penalización por distractor semántico
        if emb_dist is not None:
            simd = max(float(emb_r @ e) for e in emb_dist)
            if simd > sim:
                score *= 0.5
        # penalización por verbosidad
        if len(r.split()) > max_palabras_ok:
            score *= 0.7
        return min(score, 1.0)

    return f


def v_cobertura(keypoints, umbral: float = 0.55, max_palabras: int = 40):
    """
    Validador de COBERTURA semántica para síntesis/resumen: ¿el texto cubre cada idea
    clave (keypoint), aunque sea parafraseada? Aquí los embeddings SÍ sirven (texto
    largo, no antónimos cortos). Penaliza verbosidad por encima de max_palabras.
    Los embeddings de keypoints se calculan perezosamente (1ª llamada) y se cachean.
    """
    keypoints = [keypoints] if isinstance(keypoints, str) else list(keypoints)
    cache = {}

    def f(r: str) -> float:
        r = (r or "").strip()
        if not r:
            return 0.0
        if "kp" not in cache:
            cache["kp"] = embed(keypoints)
        emb_kp = cache["kp"]
        chunks = [c.strip() for c in re.split(r'[.\n;,]', r) if c.strip()] or [r]
        emb_ch = embed(chunks)
        cov = 0.0
        for ek in emb_kp:
            best = max(float(ek @ ec) for ec in emb_ch)
            cov += 1.0 if best >= umbral else max(0.0, (best - 0.30) / (umbral - 0.30))
        cov /= len(keypoints)
        nw = len(r.split())
        if nw > max_palabras:
            cov *= max(0.5, 1.0 - (nw - max_palabras) / max_palabras)
        return min(cov, 1.0)

    return f


if __name__ == "__main__":
    # Smoke test (CPU): correr en terminal -> evolution/bin/python validacion_semantica.py
    v = v_semantico("baja", distractores=("sube",))
    for resp, etiqueta in [("baja", "exacto"), ("cae", "sinónimo"),
                           ("desciende hacia el suelo", "sinónimo+verboso"),
                           ("sube", "distractor"), ("el clima es agradable", "irrelevante")]:
        print(f"  {etiqueta:18s} '{resp[:30]}' -> {v(resp):.2f}")
    print("similitud baja~cae:", round(similitud("baja", "cae"), 3))
    print("similitud baja~sube:", round(similitud("baja", "sube"), 3))
