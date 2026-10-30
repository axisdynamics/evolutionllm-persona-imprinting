# entorno_banco.py
# Entorno de ENTRENAMIENTO basado en el banco calibrado (presión selectiva).
#
# Implementa la misma interfaz que EntornoDificil (evaluar_poblacion + adaptar_entorno)
# para que MotorEvolutivoProfesional funcione sin cambios. Diferencias:
#   - Las tareas vienen del subset TRAIN del banco (no de las 5 fijas).
#   - Reina Roja sobre DIFICULTAD CALIBRADA: empieza con los bins fáciles y desbloquea
#     bins más difíciles a medida que la población los domina (motor llama adaptar_entorno
#     cuando fitness_avg > 0.75).
#   - Cada generación MUESTREA n_tareas_gen del rango activo (controla costo + da variedad,
#     lo que reduce varianza sin necesidad de evaluaciones=N altas).
#   - Fitness = media SIMPLE no ponderada (comparable; la ponderación causaba el artefacto).
#
# El termómetro de MEDICIÓN (held-out fijo) es aparte: medir_termometro().

import random
from collections import defaultdict
from typing import List

from organismo import OrganismoLLM
from dificultad import dificultad_continua, bin_dificultad


class EntornoBanco:
    def __init__(self, backend, banco_train: List, calib: dict, n_tareas_gen: int = 8,
                 n_bins: int = 7, bin_inicial: int = 2, max_new_tokens: int = 600, seed: int = None):
        self.backend = backend
        self.calib = calib
        self.n_bins = n_bins
        self.n_tareas_gen = n_tareas_gen
        self.max_new_tokens = max_new_tokens
        self.rng = random.Random(seed)

        # agrupar tareas train por bin de dificultad calibrada (sin calibrar -> bin 1)
        self.por_bin = defaultdict(list)
        for t in banco_train:
            p = calib.get(t.id_tarea)
            b = bin_dificultad(dificultad_continua(p), n_bins) if p is not None else 1
            self.por_bin[b].append(t)

        self.bin_activo = min(bin_inicial, n_bins)
        self.tareas = []  # tareas muestreadas en la última generación (para logging)

    def tareas_disponibles(self) -> List:
        ts = []
        for b in range(1, self.bin_activo + 1):
            ts += self.por_bin.get(b, [])
        if not ts:  # por si los bins bajos están vacíos
            ts = [t for lst in self.por_bin.values() for t in lst]
        return ts

    def _muestrear(self) -> List:
        disp = self.tareas_disponibles()
        k = min(self.n_tareas_gen, len(disp))
        return self.rng.sample(disp, k)

    # --- interfaz que usa el motor ---
    def evaluar_poblacion(self, poblacion: List[OrganismoLLM], num_evaluaciones: int,
                          ponderar_por_dificultad: bool = False):
        tareas = self._muestrear()
        self.tareas = tareas

        prompts, refs = [], []
        for org in poblacion:
            org.puntuaciones_tareas = {}
            org.dificultades_tareas = {}
            for _ in range(num_evaluaciones):
                for t in tareas:
                    prompts.append(org.compilar_prompt(t.prompt))
                    refs.append((org, t))

        print(f"🧠 Evaluando {len(prompts)} prompts ({len(tareas)} tareas, "
              f"bins 1-{self.bin_activo}) ...")
        resultados = self.backend.generate(prompts, max_new_tokens=self.max_new_tokens, batch_size=16)

        for (org, t), res in zip(refs, resultados):
            score = t.evaluar_respuesta(res['generated_text'])
            org.registrar_puntuacion(t.id_tarea, score, t.dificultad)

        # fitness = media simple NO ponderada (siempre, ignora el arg)
        for org in poblacion:
            org.calcular_fitness(ponderar_por_dificultad=False)

    def adaptar_entorno(self):
        """Reina Roja: la población dominó el rango actual -> desbloquear el siguiente bin."""
        if self.bin_activo < self.n_bins:
            self.bin_activo += 1
            n = len(self.tareas_disponibles())
            print(f"\n>> 🔴 REINA ROJA: desbloqueado bin de dificultad {self.bin_activo}/{self.n_bins} "
                  f"({n} tareas disponibles)\n")

    def get_info_tareas(self):
        return [{"id": t.id_tarea, "dif": t.dificultad} for t in self.tareas]


# ============================================================
#  TERMÓMETRO: medición en held-out (capacidad + estilo), aparte de la selección
# ============================================================
def medir_termometro(env_fijo, genotipo, calib: dict, n_bins: int = 7, generacion: int = 0) -> dict:
    """
    Mide un genotipo (el campeón) contra el held-out fijo. Devuelve fitness + perfil por
    categoría + por bin de dificultad calibrada + huella de estilo. NO afecta la selección.
    env_fijo: EntornoEvaluacionFijo(backend, K, tareas=heldout).
    """
    import re
    import statistics as st
    r = env_fijo.evaluar_genotipo(genotipo)
    porcat, porbin = defaultdict(list), defaultdict(list)
    for tid, v in r["desglose"].items():
        cat = re.match(r'(.+)_L\d+_', tid)
        porcat[cat.group(1) if cat else tid].append(v["mean"])
        p = calib.get(tid)
        b = bin_dificultad(dificultad_continua(p), n_bins) if p is not None else None
        if b is not None:
            porbin[b].append(v["mean"])
    return {
        "generacion": generacion,
        "fitness_heldout": round(r["fitness"], 4),
        "por_categoria": {k: round(st.mean(v), 3) for k, v in porcat.items()},
        "por_dificultad_bin": {b: round(st.mean(v), 3) for b, v in sorted(porbin.items())},
        "estilo": {k: round(v, 3) for k, v in r.get("estilo", {}).items()},
        "estilo_por_categoria": {
            cat: {k: round(v, 2) for k, v in idx.items()}
            for cat, idx in r.get("estilo_por_categoria", {}).items()
        },
    }
