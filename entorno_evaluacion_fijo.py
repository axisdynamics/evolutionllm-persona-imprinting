# entorno_evaluacion_fijo.py
# Termómetro de evaluación FIJO para EvolutionLLM.
#
# Propósito: medir el desempeño de un genotipo con una vara ESTABLE y comparable
# entre runs, generaciones e identidades. NO se usa para la presión selectiva
# (eso sigue siendo EntornoDificil adaptativo). Es el equivalente a un test set
# fijo en ML: nunca cambia, nunca entrena, solo mide.
#
# Diferencias clave vs EntornoDificil:
#   - Instancias de tarea CONGELADAS (no se randomizan en cada llamada).
#   - NO escala dificultad (no hay adaptar_entorno / Reina Roja).
#   - Fitness = media SIMPLE no ponderada (comparable entre generaciones).
#   - Cada tarea se evalúa K veces y se promedia (amortigua ruido del LLM).
#   - Math curada con soluciones ENTERAS (evita la fragilidad del validador float).

import random
from typing import Dict, Any, List

from organismo import Genotipo, OrganismoLLM
from entorno_dificil import (
    Tarea, TareaFactoryLogica, TareaFactoryDificil,
    CoTValidator, RhymeValidator,
)
import estilo
import re
import pyphen
from thefuzz import fuzz

SEED_ENTORNO_FIJO = 777  # congela las instancias seedeadas; NO cambiar entre runs


class EntornoEvaluacionFijo:
    """Batería fija de tareas para medir genotipos de forma comparable."""

    DEFAULT_MAX_TOKENS = 600

    def __init__(self, backend, K: int = 3, max_new_tokens: int = None, tareas=None):
        self.backend = backend
        self.K = K
        self.max_new_tokens = max_new_tokens or self.DEFAULT_MAX_TOKENS
        try:
            self._pyphen = pyphen.Pyphen(lang='es_ES')
        except Exception:
            self._pyphen = pyphen.Pyphen(lang='es')
        # tareas: si se pasa una lista (p.ej. held-out de banco_tareas), se usa esa;
        # si no, se construye la batería fija de 9 tareas por defecto.
        self.tareas = list(tareas) if tareas is not None else self._construir_bateria()

    # ---------- construcción de la batería congelada ----------
    def _contar_silabas(self, texto: str) -> int:
        palabras = re.findall(r'\w+', texto.lower())
        return sum(len(self._pyphen.inserted(p).split('-')) for p in palabras)

    def _tarea_extraccion(self) -> Tarea:
        return Tarea(
            "extraccion",
            "Párrafo: 'De los planetas rocosos, Marte es el más alejado del Sol. "
            "Sin embargo, el planeta más grande, conocido por su Gran Mancha Roja, "
            "es un gigante gaseoso.' Responde únicamente con el nombre del gigante gaseoso.",
            lambda r: (fuzz.ratio(r.strip().lower(), 'júpiter') / 100.0) /
                      (2.0 if 'marte' in r.strip().lower() else 1.0),
            dificultad=1, descripcion="Extracción NLP con penalización",
        )

    def _tarea_poesia(self) -> Tarea:
        def validar_poesia(r: str) -> float:
            score = 0.0
            versos = [v.strip() for v in r.split('\n') if v.strip()]
            if len(versos) < 3:
                return 0.0
            vv = versos[:4] if len(versos) >= 4 else versos
            if len(vv) >= 4:
                score += 0.2
            elif len(vv) >= 3:
                score += 0.1
            score += RhymeValidator.validate_abab(vv) * 0.4
            for sil in [self._contar_silabas(v) for v in vv]:
                if abs(sil - 11) <= 2:
                    score += 0.1
                elif abs(sil - 11) <= 3:
                    score += 0.05
            return min(score, 1.0)
        return Tarea(
            "poesia_metrica",
            "Crea una cuarteta (4 versos) con rima ABAB. "
            "Cada verso debe tener 11 sílabas (Endecasílabos).",
            validar_poesia, dificultad=2, descripcion="Poesía con métrica estricta",
        )

    def _tarea_math(self, id_suf: str, a: int, b: int, c: int, d: int, dificultad: int) -> Tarea:
        """Crea una tarea matemática con instancia FIJA. Exige solución entera."""
        sol = (c * d) / a - b
        assert sol == int(sol), f"instancia math no entera: {a}*(x+{b})={c}*{d} -> {sol}"
        sol = int(sol)
        prompt = f"Calcula 'x' en la ecuación {a}*(x + {b}) = {c}*{d}. Muestra los pasos para despejar x."

        def validar(r: str) -> float:
            r_clean = r.replace(',', '.')
            if str(sol) in re.findall(r'-?\d+', r_clean):
                return 1.0
            for num in re.findall(r'-?\d+\.?\d*', r_clean):
                try:
                    if abs(float(num) - sol) < 0.01:
                        return 1.0
                except ValueError:
                    continue
            return 0.0
        return Tarea(f"matematicas_{id_suf}", prompt, validar, dificultad, "Ecuación algebraica (fija)")

    def _construir_bateria(self) -> List[Tarea]:
        # Instancias seedeadas (silogismo y código usan random.sample / factorial fijo)
        random.seed(SEED_ENTORNO_FIJO)
        logica = TareaFactoryLogica()
        dificil = TareaFactoryDificil()

        cod1 = dificil.crear_tarea_codigo(dificultad=1)   # factorial(4)
        cod3 = dificil.crear_tarea_codigo(dificultad=3)   # factorial(6)
        # ids únicos: la fábrica devuelve "codigo" para ambos -> colisionan en un bucket
        cod1.id_tarea = "codigo_f4"
        cod3.id_tarea = "codigo_f6"

        tareas = [
            self._tarea_extraccion(),                       # NLP
            logica.crear_silogismo_complejo(dificultad=1),  # lógica fácil
            logica.crear_silogismo_complejo(dificultad=3),  # lógica difícil
            self._tarea_poesia(),                           # poesía/métrica
            cod1,                                           # código factorial(4)
            cod3,                                           # código factorial(6)
            # math curada, soluciones enteras, dificultad creciente:
            self._tarea_math("e", a=2, b=3, c=10, d=1, dificultad=1),  # x=2
            self._tarea_math("m", a=3, b=4, c=9,  d=5, dificultad=2),  # x=11
            self._tarea_math("d", a=4, b=5, c=8,  d=9, dificultad=3),  # x=13
        ]
        return tareas

    # ---------- evaluación ----------
    def evaluar_genotipo(self, genotipo: Genotipo, K: int = None) -> Dict[str, Any]:
        """Evalúa un genotipo K veces por tarea. Devuelve fitness (media simple) y desglose."""
        K = K or self.K
        org = OrganismoLLM(genotipo, generacion=0)

        prompts, refs = [], []
        for _ in range(K):
            for tarea in self.tareas:
                prompts.append(org.compilar_prompt(tarea.prompt))
                refs.append(tarea)

        resultados = self.backend.generate(prompts, max_new_tokens=self.max_new_tokens, batch_size=16)
        feats = []
        for res, tarea in zip(resultados, refs):
            texto = res['generated_text']
            score = tarea.evaluar_respuesta(texto)
            org.registrar_puntuacion(tarea.id_tarea, score, tarea.dificultad)
            # ESTILO: extraer del texto del modelo (mismo recorte que Tarea.evaluar_respuesta)
            if "<start_of_turn>model\n" in texto:
                texto_modelo = texto.split("<start_of_turn>model\n", 1)[1]
            else:
                texto_modelo = texto
            feats.append(estilo.extraer_features(texto_modelo))

        # estilo por categoría (índices interpretables) — para ver el estilo POR tipo de tarea
        from collections import defaultdict
        feats_cat = defaultdict(list)
        for ref, fe in zip(refs, feats):
            feats_cat[ref.descripcion].append(fe)
        estilo_por_categoria = {
            cat: estilo.indices_estilo(estilo.promediar(fl)) for cat, fl in feats_cat.items()
        }

        # fitness = media simple NO ponderada (comparable). NO incluye estilo: el estilo es
        # descriptivo (huella), no entra a la selección, para que EMERJA de la identidad.
        fitness = org.calcular_fitness(ponderar_por_dificultad=False)
        return {
            "fitness": fitness,
            "desglose": org.get_desglose_fitness(),
            "estilo": estilo.promediar(feats),                 # huella global (vector completo)
            "estilo_por_categoria": estilo_por_categoria,      # índices por categoría
        }

    def info_tareas(self) -> List[Dict[str, Any]]:
        return [{"id": t.id_tarea, "dif": t.dificultad, "desc": t.descripcion} for t in self.tareas]
