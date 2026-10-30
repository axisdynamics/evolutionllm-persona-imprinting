# motor_evolutivo.py
# VERSIÓN ROBUSTECIDA v2.1
# Correcciones: selección por torneo, pool con poda, fix bugs, guardado de campeones
# NUEVO: Filtrado de errores API, validación de genes, fallback a mutación simple

import random
import numpy as np
import copy
from typing import List, Dict, Optional, Any, Tuple
from organismo import OrganismoLLM, Genotipo
import gc
import torch
import pickle
import os
import json
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class EvolutionConfig:
    """Configuración del motor evolutivo."""
    num_evaluaciones_fitness: int = 3
    distancia_compatibilidad_especie: float = 4.0
    prob_mutacion_inteligente: float = 0.30
    max_pool_size: int = 50
    torneo_size: int = 3
    ponderar_fitness_por_dificultad: bool = True


DEFAULT_POOL_GENETICO = {
    "rol": ["experto lógico", "asistente conciso y directo", "analista de datos",
            "generador de código Python", "razonador abstracto"],
    "formato_salida": ["respuesta directa", "formato JSON", "lista de pasos numerados",
                       "código ejecutable", "explicación detallada"],
    "estrategias": ["Piensa paso a paso.", "Descompón el problema en partes más pequeñas.",
                    "Verifica tu respuesta antes de entregarla."],
    "restricciones": ["No incluyas preámbulos ni disculpas.", "Usa un lenguaje estrictamente formal.",
                      "Sé breve y ve al punto."]
}


# === PATRONES DE ERROR A FILTRAR ===
ERROR_PATTERNS = [
    "ERROR_API", "TIMEOUT", "Connection", "error", "Error",
    "failed", "Failed", "refused", "Refused",
    "<think>", "</think>", "<|", "|>",
    "Exception", "Traceback", "HTTPError",
    "ConnectionError", "TimeoutError", "refused",
    "ECONNREFUSED", "ETIMEDOUT", "socket",
    "```", "import ", "def ", "class ",  # Código en lugar de gen
    "None", "null", "undefined",
]


def is_valid_gene(gen: str) -> bool:
    """
    Valida que un gen sea válido y no contenga errores.
    Retorna True si el gen es válido, False si debe ser rechazado.
    """
    if not gen or not isinstance(gen, str):
        return False
    
    gen = gen.strip()
    
    # Longitud válida (mínimo 5, máximo 500 caracteres)
    if len(gen) < 5 or len(gen) > 500:
        return False
    
    # Verificar patrones de error
    for pattern in ERROR_PATTERNS:
        if pattern in gen:
            return False
    
    # Debe tener al menos algunos caracteres alfanuméricos
    if not any(c.isalnum() for c in gen):
        return False
    
    # No debe ser solo puntuación o espacios
    alphanumeric_ratio = sum(c.isalnum() or c.isspace() for c in gen) / len(gen)
    if alphanumeric_ratio < 0.5:
        return False
    
    return True


class PoolGenetico:
    """Pool de genes con poda automática por rendimiento."""

    def __init__(self, pool_inicial: Dict[str, List[str]], max_size: int = 50):
        self.pool = copy.deepcopy(pool_inicial)
        self.max_size = max_size
        self.gene_stats: Dict[str, Dict[str, float]] = {
            key: {gen: 0.5 for gen in genes} for key, genes in self.pool.items()
        }
        self.error_count = 0  # Contador de errores filtrados

    def agregar_gen(self, categoria: str, gen: str) -> bool:
        """Agrega un gen al pool si es válido."""
        if categoria not in self.pool:
            return False
        
        # === VALIDACIÓN DE GEN ===
        if not is_valid_gene(gen):
            self.error_count += 1
            return False
        
        # Ya existe
        if gen in self.pool[categoria]:
            return False
        
        # Podar si es necesario
        if len(self.pool[categoria]) >= self.max_size:
            self._podar_categoria(categoria)
        
        self.pool[categoria].append(gen)
        self.gene_stats[categoria][gen] = 0.5
        return True

    def registrar_exito(self, categoria: str, gen: str, fitness: float):
        if categoria in self.gene_stats and gen in self.gene_stats[categoria]:
            alpha = 0.3
            self.gene_stats[categoria][gen] = alpha * fitness + (1 - alpha) * self.gene_stats[categoria][gen]

    def _podar_categoria(self, categoria: str):
        if len(self.pool[categoria]) <= 5:
            return
        genes_con_stats = [(g, self.gene_stats[categoria].get(g, 0)) for g in self.pool[categoria]]
        genes_con_stats.sort(key=lambda x: x[1], reverse=True)
        n_mantener = max(5, int(len(genes_con_stats) * 0.8))
        self.pool[categoria] = [g[0] for g in genes_con_stats[:n_mantener]]
        print(f"🧹 Pool '{categoria}' podado a {len(self.pool[categoria])} genes")

    def obtener_gen_aleatorio(self, categoria: str) -> str:
        return random.choice(self.pool[categoria])

    def obtener_genes(self, categoria: str) -> List[str]:
        return self.pool[categoria].copy()

    def limpiar_errores(self):
        """Limpia genes inválidos que pudieran haberse colado."""
        cleaned = 0
        for categoria in self.pool:
            genes_validos = [g for g in self.pool[categoria] if is_valid_gene(g)]
            if len(genes_validos) < len(self.pool[categoria]):
                cleaned += len(self.pool[categoria]) - len(genes_validos)
                self.pool[categoria] = genes_validos
                # También limpiar stats
                self.gene_stats[categoria] = {
                    g: s for g, s in self.gene_stats[categoria].items() 
                    if g in genes_validos
                }
        if cleaned > 0:
            print(f"🧹 Limpiados {cleaned} genes inválidos del pool")
        return cleaned

    def to_dict(self) -> Dict[str, Any]:
        return {"pool": self.pool, "stats": self.gene_stats, "errors_filtered": self.error_count}


class MotorEvolutivoProfesional:
    """Motor evolutivo con selección por torneo y guardado detallado."""

    def __init__(self, tamaño_poblacion: int, tasa_mutacion: float,
                 tasa_cruce: float, num_elites: int, entorno,
                 config: Optional[EvolutionConfig] = None):
        self.tamaño_poblacion = tamaño_poblacion
        self.tasa_mutacion = tasa_mutacion
        self.tasa_cruce = tasa_cruce
        self.num_elites_global = num_elites
        self.entorno = entorno
        self.config = config or EvolutionConfig()

        self.poblacion: List[OrganismoLLM] = []
        self.generacion_actual = 0
        self.historial_metricas: List[Tuple] = []
        self.salon_de_la_fama: List[OrganismoLLM] = []
        self.historial_campeones: List[OrganismoLLM] = []
        self.pool_genetico = PoolGenetico(DEFAULT_POOL_GENETICO, self.config.max_pool_size)
        self.registro_genealogico: Dict[str, dict] = {}
        self.campeones_por_generacion: List[Dict[str, Any]] = []
        
        # Contadores de errores
        self.api_errors = 0
        self.synthesis_failures = 0

    def _seleccion_torneo(self, k: int = None) -> OrganismoLLM:
        """Selección por torneo - más presión selectiva."""
        k = k or self.config.torneo_size
        participantes = random.sample(self.poblacion, min(k, len(self.poblacion)))
        return max(participantes, key=lambda x: x.fitness)

    def _sintetizar_gen(self, genotipo_base: Genotipo, tipo_gen: str) -> Optional[str]:
        """
        Sintetiza un nuevo gen usando el LLM.
        Retorna None si hay error (en lugar de retornar el error como string).
        """
        if tipo_gen == 'rol':
            prompt = f"Tu rol actual es: '{genotipo_base.genes['rol']}'. Inventa un rol diferente pero relacionado. Responde SOLO con el nuevo rol, sin explicaciones."
        else:
            est = ', '.join(genotipo_base.genes['estrategias']) or "ninguna"
            prompt = f"Estrategias actuales: {est}. Inventa una nueva estrategia de razonamiento diferente. Responde SOLO con la estrategia, sin la palabra 'Estrategia:' al inicio, sin explicaciones."

        try:
            resp = self.entorno.backend.generate([prompt], 300, 1)[0]
            texto = resp.get('generated_text', '')
            
            # Verificar si hubo error en la respuesta
            if not texto or 'error' in texto.lower() or 'Error' in texto:
                self.api_errors += 1
                return None
            
            # Extraer candidato
            if "<start_of_turn>model" in texto:
                candidato = texto.split("<start_of_turn>model")[-1].strip()
            else:
                candidato = texto[len(prompt):].strip() if len(texto) > len(prompt) else texto.strip()
            
            # Limpiar
            candidato = candidato.replace('"', '').replace("'", "").strip("*-.").strip()
            
            # Eliminar prefijo "Estrategia:" si existe
            if candidato.lower().startswith('estrategia:'):
                candidato = candidato[11:].strip()
            elif candidato.lower().startswith('estrategia'):
                candidato = candidato[10:].strip(': ').strip()
            
            # Validar antes de retornar
            if is_valid_gene(candidato):
                return candidato
            else:
                self.synthesis_failures += 1
                return None
                
        except Exception as e:
            self.api_errors += 1
            return None

    def _mutar_genotipo(self, genotipo: Genotipo) -> Genotipo:
        """Muta un genotipo usando mutación inteligente o simple."""
        g = Genotipo.from_dict(genotipo.to_dict())

        # Intentar mutación inteligente
        if random.random() < self.config.prob_mutacion_inteligente:
            tipo = 'rol' if random.random() < 0.5 else 'estrategia'
            nuevo = self._sintetizar_gen(g, tipo)
            
            # Solo usar si la síntesis fue exitosa
            if nuevo is not None:
                cat = 'estrategias' if tipo == 'estrategia' else 'rol'
                if self.pool_genetico.agregar_gen(cat, nuevo):
                    if tipo == 'rol':
                        g.genes['rol'] = nuevo
                    elif nuevo not in g.genes['estrategias']:
                        g.genes['estrategias'].append(nuevo)
                    return g
            # Si falla, continuar con mutación simple (fallback)

        # Mutación simple (siempre funciona)
        gen_a_mutar = random.choice(list(g.genes.keys()))
        if gen_a_mutar in ["rol", "formato_salida"]:
            g.genes[gen_a_mutar] = self.pool_genetico.obtener_gen_aleatorio(gen_a_mutar)
        elif gen_a_mutar in ["estrategias", "restricciones"]:
            if g.genes[gen_a_mutar] and random.random() < 0.5:
                g.genes[gen_a_mutar].pop(random.randrange(len(g.genes[gen_a_mutar])))
            else:
                nuevo = self.pool_genetico.obtener_gen_aleatorio(gen_a_mutar)
                if nuevo not in g.genes[gen_a_mutar]:
                    g.genes[gen_a_mutar].append(nuevo)
        return g

    def _crear_genotipo_aleatorio(self) -> Genotipo:
        return Genotipo(
            rol=self.pool_genetico.obtener_gen_aleatorio("rol"),
            formato_salida=self.pool_genetico.obtener_gen_aleatorio("formato_salida"),
            estrategias=random.sample(self.pool_genetico.obtener_genes("estrategias"), random.randint(0, 2)),
            restricciones=random.sample(self.pool_genetico.obtener_genes("restricciones"), random.randint(0, 2))
        )

    def _cruzar_genotipos(self, g1: Genotipo, g2: Genotipo) -> Genotipo:
        return Genotipo(
            rol=random.choice([g1.genes["rol"], g2.genes["rol"]]),
            formato_salida=random.choice([g1.genes["formato_salida"], g2.genes["formato_salida"]]),
            estrategias=random.choice([g1.genes["estrategias"].copy(), g2.genes["estrategias"].copy()]),
            restricciones=random.choice([g1.genes["restricciones"].copy(), g2.genes["restricciones"].copy()])
        )

    def _distancia_genotipica(self, g1: Genotipo, g2: Genotipo) -> float:
        d = 0
        if g1.genes["rol"] != g2.genes["rol"]: d += 1
        if g1.genes["formato_salida"] != g2.genes["formato_salida"]: d += 1
        d += len(set(g1.genes["estrategias"]).symmetric_difference(set(g2.genes["estrategias"])))
        d += len(set(g1.genes["restricciones"]).symmetric_difference(set(g2.genes["restricciones"])))
        return d

    def iniciar_poblacion(self):
        self.poblacion = [OrganismoLLM(self._crear_genotipo_aleatorio(), generacion=0)
                          for _ in range(self.tamaño_poblacion)]
        for org in self.poblacion:
            self.registro_genealogico[org.id] = {'p1': None, 'p2': None, 'gen': 0, 'fitness': 0.0}
        print(f"✅ Población inicial de {self.tamaño_poblacion} organismos creada.")

    def guardar_estado(self, ruta_directorio: str):
        os.makedirs(ruta_directorio, exist_ok=True)

        # Limpiar pool antes de guardar
        self.pool_genetico.limpiar_errores()

        # Pickle
        ruta_pkl = os.path.join(ruta_directorio, "checkpoint_motor.pkl")
        entorno_temp = self.entorno
        self.entorno = None
        with open(ruta_pkl, 'wb') as f:
            pickle.dump(self, f)
        self.entorno = entorno_temp

        # JSONs
        with open(os.path.join(ruta_directorio, "linaje.json"), 'w', encoding='utf-8') as f:
            json.dump(self.registro_genealogico, f, indent=2, ensure_ascii=False)

        with open(os.path.join(ruta_directorio, "campeones_por_generacion.json"), 'w', encoding='utf-8') as f:
            json.dump(self.campeones_por_generacion, f, indent=2, ensure_ascii=False)

        with open(os.path.join(ruta_directorio, "pool_genetico.json"), 'w', encoding='utf-8') as f:
            json.dump(self.pool_genetico.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"💾 Estado guardado en {ruta_directorio}")

    @classmethod
    def cargar_estado(cls, ruta_pickle: str, entorno_activo) -> 'MotorEvolutivoProfesional':
        with open(ruta_pickle, 'rb') as f:
            motor = pickle.load(f)
        motor.entorno = entorno_activo
        
        # Limpiar errores que pudieran estar en el estado guardado
        motor.pool_genetico.limpiar_errores()
        
        print(f"♻️ Estado cargado: Generación {motor.generacion_actual}")
        return motor

    def evolucionar(self) -> List[OrganismoLLM]:
        print(f"\n{'='*60}\n🧬 GENERACIÓN {self.generacion_actual}\n{'='*60}")

        # Evaluar
        self.entorno.evaluar_poblacion(
            self.poblacion,
            self.config.num_evaluaciones_fitness,
            ponderar_por_dificultad=self.config.ponderar_fitness_por_dificultad
        )

        # Actualizar stats del pool (solo para genes válidos)
        for org in self.poblacion:
            if is_valid_gene(org.genotipo.genes["rol"]):
                self.pool_genetico.registrar_exito("rol", org.genotipo.genes["rol"], org.fitness)
            for est in org.genotipo.genes["estrategias"]:
                if is_valid_gene(est):
                    self.pool_genetico.registrar_exito("estrategias", est, org.fitness)
            if org.id in self.registro_genealogico:
                self.registro_genealogico[org.id]['fitness'] = org.fitness

        # Ordenar
        self.poblacion.sort(key=lambda x: x.fitness, reverse=True)

        # Campeón
        campeon = self.poblacion[0]
        self.historial_campeones.append(campeon.clone())
        self.campeones_por_generacion.append({
            "generacion": self.generacion_actual,
            "id": campeon.id,
            "fitness": campeon.fitness,
            "genotipo": campeon.genotipo.to_dict()
        })

        # Salón de la fama
        self.salon_de_la_fama.append(campeon.clone())
        self.salon_de_la_fama.sort(key=lambda x: x.fitness, reverse=True)
        self.salon_de_la_fama = self.salon_de_la_fama[:self.tamaño_poblacion]

        # Métricas
        fitness_max = campeon.fitness
        fitness_avg = np.mean([o.fitness for o in self.poblacion])

        # Especiación
        especies = []
        for org in self.poblacion:
            found = False
            for esp in especies:
                if self._distancia_genotipica(org.genotipo, esp[0].genotipo) < self.config.distancia_compatibilidad_especie:
                    esp.append(org)
                    found = True
                    break
            if not found:
                especies.append([org])

        # Diversidad
        distancias = [self._distancia_genotipica(o1.genotipo, o2.genotipo)
                      for i, o1 in enumerate(self.poblacion) for o2 in self.poblacion[i+1:]]
        diversidad = np.mean(distancias) if distancias else 0

        self.historial_metricas.append((self.generacion_actual, fitness_max, fitness_avg, diversidad, len(especies)))

        print(f"📊 FMax={fitness_max:.4f} | FAvg={fitness_avg:.4f} | Div={diversidad:.2f} | Especies={len(especies)}")
        print(f"🏆 Campeón: {campeon.genotipo.genes}")
        
        # Mostrar errores si los hay
        if self.api_errors > 0 or self.synthesis_failures > 0:
            print(f"⚠️  Errores filtrados: API={self.api_errors}, Síntesis={self.synthesis_failures}")

        # Adaptar entorno si población es muy buena
        if fitness_avg > 0.75 and hasattr(self.entorno, 'adaptar_entorno'):
            self.entorno.adaptar_entorno()

        # Nueva generación
        siguiente = [elite.clone() for elite in self.salon_de_la_fama[:self.num_elites_global]]

        while len(siguiente) < self.tamaño_poblacion:
            # Selección por torneo
            padre1 = self._seleccion_torneo()
            padre2_id = None

            if random.random() < self.tasa_cruce and len(self.poblacion) > 1:
                padre2 = self._seleccion_torneo()
                padre2_id = padre2.id
                genotipo_hijo = self._cruzar_genotipos(padre1.genotipo, padre2.genotipo)
            else:
                genotipo_hijo = Genotipo.from_dict(padre1.genotipo.to_dict())

            if random.random() < self.tasa_mutacion:
                genotipo_hijo = self._mutar_genotipo(genotipo_hijo)

            # Crear hijo con generación correcta
            hijo = OrganismoLLM(genotipo_hijo, generacion=self.generacion_actual + 1,
                               padre1_id=padre1.id, padre2_id=padre2_id)
            siguiente.append(hijo)

            self.registro_genealogico[hijo.id] = {
                'p1': padre1.id, 'p2': padre2_id,
                'gen': self.generacion_actual + 1, 'fitness': 0.0
            }

        self.poblacion = siguiente
        self.generacion_actual += 1

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.poblacion