# dificultad.py
# Dificultad CONTINUA y calibrada por substrato.
#
# En vez de comprometerse a 5/7/11 niveles fijos, la dificultad de cada tarea es una
# variable continua = 1 - p_i, donde p_i es la tasa de acierto del modelo base medida
# por calibrar_banco.py (POR SUBSTRATO; ver memoria 'calibracion-por-substrato').
# El binning (para reportar/Reina Roja) se decide al momento, con cualquier n_bins.
#
# El "Ln" en el id de cada tarea queda como etiqueta de instancia, NO como dificultad.

import json
import re
import random
import statistics as st
from pathlib import Path
from collections import defaultdict

_DIR = Path(__file__).resolve().parent


def _ruta_calibracion(modelo: str) -> Path:
    return _DIR / "artifacts" / f"calibracion_{modelo.replace(':', '_').replace('/', '_')}.json"


def cargar_calibracion(modelo: str) -> dict:
    """Devuelve {task_id: p_i} para el substrato, o {} si no hay calibración aún."""
    f = _ruta_calibracion(modelo)
    if not f.exists():
        return {}
    d = json.load(open(f))
    return {t["id"]: t["p"] for t in d.get("tareas", [])}


def dificultad_continua(p):
    """p (tasa de acierto) -> dificultad en [0,1] (0=trivial, 1=imposible)."""
    return None if p is None else round(1.0 - p, 3)


def bin_dificultad(dif, n_bins: int = 7):
    """Dificultad continua [0,1] -> bin 1..n_bins (None si sin calibrar)."""
    if dif is None:
        return None
    return min(int(dif * n_bins) + 1, n_bins)


def categoria_de(task_id: str) -> str:
    m = re.match(r'(.+)_L\d+_', task_id)
    return m.group(1) if m else task_id


def anotar_banco(banco, calib: dict):
    """Adjunta .p_calibrado y .dificultad_continua a cada Tarea (in place)."""
    for t in banco:
        p = calib.get(t.id_tarea)
        t.p_calibrado = p
        t.dificultad_continua = dificultad_continua(p)
    return banco


def particion_por_dificultad(banco, calib: dict, frac_heldout: float = 0.4,
                             n_bins: int = 7, seed: int = 1234):
    """
    Partición train/held-out estratificada por (categoría × bin de dificultad CALIBRADA),
    para que el held-out abarque todo el rango de dificultad real. Si una tarea no tiene
    calibración, cae a un grupo 'sin_calibrar'.
    """
    rng = random.Random(seed)
    grupos = defaultdict(list)
    for t in banco:
        p = calib.get(t.id_tarea)
        b = bin_dificultad(dificultad_continua(p), n_bins) if p is not None else 0
        grupos[(categoria_de(t.id_tarea), b)].append(t)

    train, held = [], []
    for items in grupos.values():
        items = items[:]
        rng.shuffle(items)
        nh = max(1, round(len(items) * frac_heldout)) if len(items) > 1 else 0
        held += items[:nh]
        train += items[nh:]
    return train, held


def resumen_por_bin(pares, n_bins: int = 7) -> dict:
    """pares: [(dificultad_continua, valor)] -> {bin: media} (escalera monótona)."""
    acc = defaultdict(list)
    for dif, v in pares:
        b = bin_dificultad(dif, n_bins)
        if b is not None:
            acc[b].append(v)
    return {b: round(st.mean(vs), 3) for b, vs in sorted(acc.items())}


def umbral_para_desbloquear(banco, calib, dominio_actual: float, n_bins: int = 7):
    """
    Reina Roja auto-calibrante: dado el desempeño medio actual de la población
    (dominio_actual en [0,1]), devuelve las tareas cuya dificultad está apenas por
    encima de ese dominio (la 'frontera' donde toca empujar). Para presión selectiva.
    """
    frontera = []
    for t in banco:
        p = calib.get(t.id_tarea)
        dif = dificultad_continua(p)
        if dif is None:
            continue
        # tareas en la banda inmediatamente más difícil que el dominio actual
        if dominio_actual <= (1 - dif) < dominio_actual + (1.0 / n_bins) * 2:
            frontera.append(t)
    return frontera
