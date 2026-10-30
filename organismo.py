# organismo.py
# VERSIÓN HÍBRIDA: Base del usuario + métodos de compatibilidad
# Mantiene la estructura original pero agrega funcionalidad necesaria

import uuid
import numpy as np
from typing import List, Dict, Optional, Any
import copy


class Genotipo:
    def __init__(self, rol: str, formato_salida: str, estrategias: List[str], restricciones: List[str]):
        self.genes: Dict[str, Any] = {
            "rol": rol,
            "formato_salida": formato_salida,
            "estrategias": sorted(list(set(estrategias))),
            "restricciones": sorted(list(set(restricciones)))
        }

    def compilar_prompt(self, prompt_tarea: str) -> str:
        instrucciones = f"Rol principal: {self.genes['rol']}.\n"
        instrucciones += f"Formato de salida requerido: {self.genes['formato_salida']}.\n"
        if self.genes['estrategias']:
            instrucciones += "Estrategias a seguir:\n" + "\n".join(f"- {e}" for e in self.genes['estrategias'])
        if self.genes['restricciones']:
            instrucciones += "\nRestricciones importantes:\n" + "\n".join(f"- {r}" for r in self.genes['restricciones'])
        instruccion_final = "Basado en las instrucciones y la tarea, proporciona la respuesta directamente sin hacer preguntas."
        return (f"<start_of_turn>user\n{instrucciones.strip()}\n\n"
                f"--- TAREA ---\n{prompt_tarea}\n\n"
                f"--- INSTRUCCIÓN FINAL ---\n{instruccion_final}"
                f"<end_of_turn>\n<start_of_turn>model\n")

    # === MÉTODOS AGREGADOS PARA COMPATIBILIDAD ===
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa el genotipo a diccionario (para JSON)."""
        return copy.deepcopy(self.genes)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Genotipo':
        """Deserializa un genotipo desde diccionario."""
        return cls(
            rol=data.get("rol", ""),
            formato_salida=data.get("formato_salida", ""),
            estrategias=data.get("estrategias", []),
            restricciones=data.get("restricciones", [])
        )

    def __str__(self):
        return str(self.genes)
    
    def __repr__(self):
        return f"Genotipo({self.genes})"


class OrganismoLLM:
    def __init__(self, genotipo: Genotipo, generacion: int = 0,
                 padre1_id: Optional[str] = None, padre2_id: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.genotipo = genotipo
        self.generacion = generacion
        self.fitness = 0.0
        self.puntuaciones_tareas: Dict[str, List[float]] = {}
        
        # === ATRIBUTOS AGREGADOS PARA COMPATIBILIDAD ===
        self.dificultades_tareas: Dict[str, int] = {}  # Para ponderación por dificultad
        self.padre1_id = padre1_id  # Para tracking de linaje
        self.padre2_id = padre2_id

    def __str__(self):
        return (f"Organismo(ID: {self.id[:8]}, Gen: {self.generacion}, "
                f"Fitness: {self.fitness:.4f}, Genotipo: '{self.genotipo}')")

    def clone(self):
        cloned_genotipo = copy.deepcopy(self.genotipo)
        clon = OrganismoLLM(
            genotipo=cloned_genotipo, 
            generacion=self.generacion,
            padre1_id=self.padre1_id,
            padre2_id=self.padre2_id
        )
        clon.id = self.id  # Mantener mismo ID para élites
        clon.fitness = self.fitness
        clon.puntuaciones_tareas = copy.deepcopy(self.puntuaciones_tareas)
        clon.dificultades_tareas = copy.deepcopy(self.dificultades_tareas)
        return clon

    def to_dict(self) -> Dict[str, Any]:
        """Serializa el organismo completo (mejorado para JSON)."""
        return {
            "id": self.id,
            "fitness": self.fitness,
            "generacion": self.generacion,
            "genotipo": self.genotipo.to_dict(),  # Ahora es dict, no string
            "padre1_id": self.padre1_id,
            "padre2_id": self.padre2_id
        }
        
    def compilar_prompt(self, prompt_tarea: str) -> str:
        return self.genotipo.compilar_prompt(prompt_tarea)

    # === MÉTODO ORIGINAL (mantenido para compatibilidad hacia atrás) ===
    def calcular_fitness(self, ponderar_por_dificultad: bool = False) -> float:
        """
        Calcula el fitness del organismo.
        
        Args:
            ponderar_por_dificultad: Si True, tareas más difíciles pesan más.
                                     Si False (default), usa media simple (comportamiento original).
        
        Returns:
            Fitness calculado
        """
        if not self.puntuaciones_tareas:
            self.fitness = 0.0
            return self.fitness
        
        if ponderar_por_dificultad and self.dificultades_tareas:
            # Promedio ponderado por dificultad
            suma_ponderada = 0.0
            suma_pesos = 0.0
            
            for tarea_id, scores in self.puntuaciones_tareas.items():
                if scores:
                    mean_score = np.mean(scores)
                    dificultad = self.dificultades_tareas.get(tarea_id, 1)
                    suma_ponderada += mean_score * dificultad
                    suma_pesos += dificultad
            
            self.fitness = suma_ponderada / suma_pesos if suma_pesos > 0 else 0.0
        else:
            # Comportamiento original: media simple
            todas_las_puntuaciones = [
                score for scores in self.puntuaciones_tareas.values() 
                for score in scores
            ]
            self.fitness = np.mean(todas_las_puntuaciones) if todas_las_puntuaciones else 0.0
        
        return self.fitness

    # === MÉTODOS AGREGADOS PARA COMPATIBILIDAD ===
    
    def registrar_puntuacion(self, tarea_id: str, puntuacion: float, dificultad: int = 1):
        """
        Registra una puntuación para una tarea específica.
        
        Args:
            tarea_id: Identificador de la tarea
            puntuacion: Score obtenido [0.0, 1.0]
            dificultad: Nivel de dificultad de la tarea
        """
        if tarea_id not in self.puntuaciones_tareas:
            self.puntuaciones_tareas[tarea_id] = []
            self.dificultades_tareas[tarea_id] = dificultad
        
        self.puntuaciones_tareas[tarea_id].append(puntuacion)
    
    def get_desglose_fitness(self) -> Dict[str, Dict[str, float]]:
        """
        Retorna un desglose detallado del fitness por tarea.
        Útil para análisis post-hoc.
        """
        desglose = {}
        for tarea_id, scores in self.puntuaciones_tareas.items():
            if scores:
                desglose[tarea_id] = {
                    "mean": float(np.mean(scores)),
                    "min": float(min(scores)),
                    "max": float(max(scores)),
                    "n_evaluaciones": len(scores),
                    "dificultad": self.dificultades_tareas.get(tarea_id, 1)
                }
        return desglose
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrganismoLLM':
        """Deserializa un organismo desde diccionario."""
        genotipo = Genotipo.from_dict(data.get("genotipo", {}))
        org = cls(
            genotipo=genotipo,
            generacion=data.get("generacion", 0),
            padre1_id=data.get("padre1_id"),
            padre2_id=data.get("padre2_id")
        )
        org.id = data.get("id", str(uuid.uuid4()))
        org.fitness = data.get("fitness", 0.0)
        return org
