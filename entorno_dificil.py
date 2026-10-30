# entorno_dificil.py
# VERSIÓN ROBUSTECIDA
# Correcciones aplicadas:
# 1. Validación CoT más robusta (no solo strings literales)
# 2. Sandbox seguro para ejecución de código
# 3. Validación de rima mejorada (análisis fonético)
# 4. max_new_tokens configurable
# 5. Registro de dificultad en puntuaciones

import re
import math
import random
import numpy as np
from thefuzz import fuzz
from llm_backends import LLMBackend
from organismo import OrganismoLLM
import pyphen
from typing import List, Callable, Optional, Dict, Any
import ast
import sys
from io import StringIO
import signal
import threading
from contextlib import contextmanager


@contextmanager
def limite_tiempo(segundos: float):
    """
    Corta cualquier bloque que exceda 'segundos' (SIGALRM, Linux, hilo principal).
    Imprescindible para ejecutar código generado por el LLM: un `while True:` colgaría
    el proceso para siempre. Fuera del hilo principal hace no-op (signals solo en main).
    """
    en_main = threading.current_thread() is threading.main_thread()
    if hasattr(signal, "setitimer") and en_main:
        def _handler(signum, frame):
            raise TimeoutError(f"excedió {segundos}s")
        old = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, segundos)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    else:
        yield


# --- Sandbox Seguro para Ejecución de Código ---
class SecureSandbox:
    """
    Sandbox para ejecutar código Python de forma segura.
    Restringe acceso a funciones peligrosas y limita tiempo de ejecución.
    """
    
    # Funciones prohibidas
    FORBIDDEN_NAMES = {
        'exec', 'eval', 'compile', 'open', 'input', '__import__',
        'getattr', 'setattr', 'delattr', 'globals', 'locals',
        'vars', 'dir', 'type', 'isinstance', 'issubclass',
        'breakpoint', 'help', 'exit', 'quit'
    }
    
    # Módulos prohibidos
    FORBIDDEN_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'socket', 'requests',
        'urllib', 'http', 'ftplib', 'smtplib', 'pickle', 'shelve',
        'importlib', 'builtins', '__builtins__'
    }

    # Módulos de CÓMPUTO permitidos (sin I/O ni acceso al sistema). Permite que
    # soluciones legítimas con imports (lru_cache, heapq, etc.) no se penalicen.
    SAFE_MODULES = {
        'math', 'functools', 'heapq', 'collections', 'itertools', 'bisect',
        're', 'string', 'fractions', 'decimal', 'random', 'statistics', 'operator',
    }
    
    @classmethod
    def validate_code(cls, code: str) -> bool:
        """
        Valida que el código no contenga operaciones peligrosas.
        Retorna True si es seguro, False si no.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        
        for node in ast.walk(tree):
            # Verificar llamadas a funciones prohibidas
            if isinstance(node, ast.Name) and node.id in cls.FORBIDDEN_NAMES:
                return False
            
            # Verificar imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split('.')[0] in cls.FORBIDDEN_MODULES:
                            return False
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split('.')[0] in cls.FORBIDDEN_MODULES:
                        return False
            
            # Verificar acceso a atributos peligrosos
            if isinstance(node, ast.Attribute):
                if node.attr.startswith('_'):
                    return False
        
        return True
    
    @classmethod
    def execute(cls, code: str, timeout_seconds: float = 2.0) -> Dict[str, Any]:
        """
        Ejecuta código en un namespace restringido con timeout.
        
        Returns:
            Dict con 'success', 'namespace', 'stdout', 'error'
        """
        result = {
            'success': False,
            'namespace': {},
            'stdout': '',
            'error': None
        }
        
        # Validar código primero
        if not cls.validate_code(code):
            result['error'] = "Código contiene operaciones prohibidas"
            return result
        
        # Crear namespace seguro
        safe_namespace = {
            '__builtins__': {
                'range': range,
                'len': len,
                'int': int,
                'float': float,
                'str': str,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'bool': bool,
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'sorted': sorted,
                'reversed': reversed,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'print': print,
                'round': round, 'pow': pow, 'divmod': divmod, 'all': all, 'any': any,
                'frozenset': frozenset, 'ord': ord, 'chr': chr, 'repr': repr,
                'format': format, 'slice': slice, 'next': next, 'iter': iter,
                'hash': hash, 'complex': complex, 'bytes': bytes,
                'True': True,
                'False': False,
                'None': None,
            }
        }

        # Import guardado: solo módulos de cómputo de la whitelist
        def _guarded_import(name, *a, **k):
            if name.split('.')[0] in cls.SAFE_MODULES:
                return __import__(name, *a, **k)
            raise ImportError(f"import '{name}' bloqueado en sandbox")
        safe_namespace['__builtins__']['__import__'] = _guarded_import
        
        # Capturar stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        # Timeout REAL: mata código colgado (p.ej. `while True:` a nivel de módulo).
        # OJO: esto solo cubre la DEFINICIÓN; la LLAMADA a la función debe envolverse
        # con limite_tiempo() donde se invoque (ver v_codigo en banco_tareas.py).
        try:
            with limite_tiempo(timeout_seconds):
                exec(code, safe_namespace)
            result['success'] = True
            result['namespace'] = {k: v for k, v in safe_namespace.items()
                                   if not k.startswith('_')}
            result['stdout'] = sys.stdout.getvalue()
        except Exception as e:
            result['error'] = str(e)
        finally:
            sys.stdout = old_stdout

        return result


# --- Validadores CoT Mejorados ---
class CoTValidator:
    """
    Validador robusto para Chain-of-Thought.
    No solo busca strings literales, sino patrones de razonamiento.
    """
    
    # Patrones que indican razonamiento secuencial
    SEQUENTIAL_PATTERNS = [
        r'\b(paso|step)\s*\d+',  # "paso 1", "step 2"
        r'\b(primero|first|segundo|second|tercero|third)\b',
        r'\b(luego|entonces|después|therefore|then|next)\b',
        r'\b(por lo tanto|por ende|en consecuencia|thus|hence)\b',
        r'\b(dado que|since|because|ya que|puesto que)\b',
        r'\b(esto significa|this means|implica que|implies)\b',
        r'\b(conclu[iy]|conclusion|resultado|result)\w*\b',
    ]
    
    # Patrones que indican análisis lógico
    LOGIC_PATTERNS = [
        r'\b(si|if)\s+.+\b(entonces|then|,)\b',  # Condicionales
        r'\b(todos|all|ningún|ninguno|none|no)\s+\w+\s+(son|are|es|is)\b',
        r'\b(contradicción|contradiction|imposible|impossible)\b',
        r'\b(válido|valid|inválido|invalid|verdadero|true|falso|false)\b',
    ]
    
    # Patrones estructurales (formato de razonamiento)
    STRUCTURE_PATTERNS = [
        r'^\s*[-•]\s+',  # Bullets
        r'^\s*\d+[.)]\s+',  # Listas numeradas
        r'\n\s*\n',  # Párrafos separados (indica estructura)
    ]
    
    @classmethod
    def validate(cls, response: str, expected_conclusion: str = "no") -> float:
        """
        Valida la calidad del Chain-of-Thought.
        
        Args:
            response: Respuesta del modelo
            expected_conclusion: Conclusión esperada ("no", "sí", etc.)
        
        Returns:
            Score [0.0, 1.0]
        """
        score = 0.0
        r_lower = response.lower()
        
        # 1. Evidencia de razonamiento secuencial (0.35 pts)
        sequential_matches = sum(
            1 for pattern in cls.SEQUENTIAL_PATTERNS 
            if re.search(pattern, r_lower, re.IGNORECASE)
        )
        if sequential_matches >= 3:
            score += 0.35
        elif sequential_matches >= 2:
            score += 0.25
        elif sequential_matches >= 1:
            score += 0.15
        
        # 2. Evidencia de análisis lógico (0.25 pts)
        logic_matches = sum(
            1 for pattern in cls.LOGIC_PATTERNS 
            if re.search(pattern, r_lower, re.IGNORECASE)
        )
        if logic_matches >= 2:
            score += 0.25
        elif logic_matches >= 1:
            score += 0.15
        
        # 3. Estructura del razonamiento (0.15 pts)
        structure_matches = sum(
            1 for pattern in cls.STRUCTURE_PATTERNS 
            if re.search(pattern, response, re.MULTILINE)
        )
        if structure_matches >= 2:
            score += 0.15
        elif structure_matches >= 1:
            score += 0.10
        
        # 4. Conclusión correcta (0.25 pts)
        # Buscar la conclusión en diferentes formatos
        conclusion_patterns = [
            rf'(conclusi[óo]n|respuesta|resultado|answer|result)[:\s]+.*\b{expected_conclusion}\b',
            rf'\b{expected_conclusion}\b[.!]?\s*$',  # Al final
            rf'(es|is)\s+(que\s+)?{expected_conclusion}\b',
        ]
        
        for pattern in conclusion_patterns:
            if re.search(pattern, r_lower, re.IGNORECASE | re.MULTILINE):
                score += 0.25
                break
        
        return min(score, 1.0)
    
    @classmethod
    def detect_marker_stuffing(cls, response: str) -> bool:
        """
        Detecta si el modelo está "hackeando" insertando marcadores
        sin razonamiento real (repetición excesiva de marcadores).
        """
        markers = ["paso", "primero", "luego", "entonces", "por lo tanto"]
        marker_count = sum(response.lower().count(m) for m in markers)
        word_count = len(response.split())
        
        # Si más del 20% de las palabras son marcadores, es sospechoso
        if word_count > 0 and marker_count / word_count > 0.2:
            return True
        return False


# --- Validador de Rima Mejorado ---
class RhymeValidator:
    """
    Validador de rima más sofisticado usando análisis fonético.
    """
    
    # Mapeo de terminaciones que riman (español)
    RHYME_GROUPS = {
        # Vocales tónicas
        'a': ['a', 'á'],
        'e': ['e', 'é'],
        'i': ['i', 'í', 'y'],
        'o': ['o', 'ó'],
        'u': ['u', 'ú'],
        # Consonantes comunes
        'ar': ['ar', 'ár'],
        'er': ['er', 'ér'],
        'ir': ['ir', 'ír'],
        'or': ['or', 'ór'],
        'on': ['on', 'ón'],
        'an': ['an', 'án'],
        'en': ['en', 'én'],
        'ado': ['ado'],
        'ido': ['ido'],
        'ción': ['ción', 'sión'],
        'dad': ['dad', 'tad'],
        'mente': ['mente'],
    }
    
    @classmethod
    def get_rhyme_ending(cls, word: str, min_chars: int = 2, max_chars: int = 4) -> str:
        """Extrae la terminación relevante para rima."""
        word = word.lower().strip('.,;:!?¿¡"\'')
        
        # Intentar encontrar terminación conocida
        for length in range(max_chars, min_chars - 1, -1):
            if len(word) >= length:
                ending = word[-length:]
                for group, variants in cls.RHYME_GROUPS.items():
                    if ending in variants:
                        return group
        
        # Fallback: últimos caracteres
        return word[-min_chars:] if len(word) >= min_chars else word
    
    @classmethod
    def words_rhyme(cls, word1: str, word2: str) -> bool:
        """Determina si dos palabras riman."""
        ending1 = cls.get_rhyme_ending(word1)
        ending2 = cls.get_rhyme_ending(word2)
        
        # Comparación directa
        if ending1 == ending2:
            return True
        
        # Verificar si están en el mismo grupo de rima
        for group, variants in cls.RHYME_GROUPS.items():
            if ending1 in variants and ending2 in variants:
                return True
        
        # Comparación fonética aproximada (últimas 2-3 letras)
        if len(word1) >= 2 and len(word2) >= 2:
            if word1[-2:] == word2[-2:]:
                return True
        
        return False
    
    @classmethod
    def validate_abab(cls, verses: List[str]) -> float:
        """
        Valida patrón de rima ABAB.
        Returns score [0.0, 1.0]
        """
        if len(verses) < 4:
            return 0.0
        
        # Extraer última palabra de cada verso
        last_words = []
        for v in verses[:4]:
            words = re.findall(r'\w+', v)
            if words:
                last_words.append(words[-1])
        
        if len(last_words) < 4:
            return 0.0
        
        score = 0.0
        
        # Verificar rima A (versos 1 y 3)
        if cls.words_rhyme(last_words[0], last_words[2]):
            score += 0.5
        
        # Verificar rima B (versos 2 y 4)
        if cls.words_rhyme(last_words[1], last_words[3]):
            score += 0.5
        
        return score


# --- Clase Tarea ---
class Tarea:
    def __init__(self, id_tarea: str, prompt: str, funcion_validacion: Callable, 
                 dificultad: int = 1, descripcion: str = ""):
        self.id_tarea = id_tarea
        self.prompt = prompt
        self.funcion_validacion = funcion_validacion
        self.dificultad = dificultad
        self.descripcion = descripcion or id_tarea

    def evaluar_respuesta(self, respuesta_completa: str) -> float:
        try:
            # Extraer solo la respuesta del modelo
            if "<start_of_turn>model\n" in respuesta_completa:
                texto_generado = respuesta_completa.split("<start_of_turn>model\n", 1)[1].strip()
            else:
                texto_generado = respuesta_completa
            
            return self.funcion_validacion(texto_generado)
        except Exception as e:
            print(f"⚠️ Error evaluando tarea {self.id_tarea}: {e}")
            return 0.0


# --- Fábricas de Tareas ---
class TareaFactoryLogica:
    """Fábrica de tareas de razonamiento lógico (Chain-of-Thought)."""
    
    def crear_silogismo_complejo(self, dificultad: int = 1) -> Tarea:
        entidades = ["Gorks", "Morks", "Zorks", "Ploogs", "Blergs", "Flimps", "Twerks"]
        
        # Más entidades para mayor dificultad
        n_entidades = min(3 + dificultad, len(entidades))
        seleccion = random.sample(entidades, n_entidades)
        entidad_A, entidad_B = seleccion[0], seleccion[1]
        entidad_C = seleccion[2] if len(seleccion) > 2 else seleccion[0]
        
        prompt = (
            f"Premisa 1: Todos los {entidad_A} son {entidad_B}.\n"
            f"Premisa 2: Ningún {entidad_B} es {entidad_C}.\n"
            f"Pregunta: ¿Es posible que exista un {entidad_A} que sea {entidad_C}? "
            "Razona tu respuesta paso a paso antes de concluir."
        )
        
        def validar_cot(r: str) -> float:
            # Detectar marker stuffing (hackeo)
            if CoTValidator.detect_marker_stuffing(r):
                return 0.1  # Penalizar fuertemente
            
            return CoTValidator.validate(r, expected_conclusion="no")
        
        return Tarea(
            f"razonamiento_cot_silogismo_dif{dificultad}", 
            prompt, 
            validar_cot, 
            dificultad,
            "Silogismo lógico con Chain-of-Thought"
        )


class TareaFactoryDificil:
    """Fábrica de tareas de dificultad variable."""
    
    def crear_tarea_matematica(self, dificultad: int = 1) -> Tarea:
        a = random.randint(1, 5)
        b = random.randint(1, 5) * dificultad
        c = random.randint(5, 10) * dificultad
        d = random.randint(1, 5) * dificultad
        
        prompt = f"Calcula 'x' en la ecuación {a}*(x + {b}) = {c}*{d}. Muestra los pasos para despejar x."
        solucion_final = (c * d) / a - b
        
        def validar(r: str) -> float:
            val_float = round(solucion_final, 2)
            val_int = int(solucion_final) if solucion_final == int(solucion_final) else None
            
            r_clean = r.replace(',', '.')
            
            # Buscar el número en la respuesta
            if str(val_float) in r_clean:
                return 1.0
            if val_int is not None and str(val_int) in r_clean:
                return 1.0
            
            # Buscar con regex para números cercanos
            numeros = re.findall(r'-?\d+\.?\d*', r_clean)
            for num in numeros:
                try:
                    if abs(float(num) - solucion_final) < 0.01:
                        return 1.0
                except ValueError:
                    continue
            
            return 0.0
        
        return Tarea("matematicas", prompt, validar, dificultad, "Ecuación algebraica")
    
    def crear_tarea_codigo(self, dificultad: int = 1) -> Tarea:
        n = 3 + dificultad
        solucion = math.factorial(n)
        prompt = (
            f"Escribe una función de Python recursiva llamada 'factorial' "
            f"que calcule el factorial de 'n'. Demuéstrala calculando el factorial de {n}."
        )
        
        def validar(r: str) -> float:
            score = 0.0
            
            # Extraer bloque de código
            if "```python" in r:
                r_code = r.split("```python", 1)[1].split("```")[0]
            elif "```" in r:
                r_code = r.split("```", 1)[1].split("```")[0]
            else:
                r_code = r
            
            # Análisis estático (0.5 pts)
            if 'def factorial' in r_code:
                score += 0.15
            if re.search(r'if\s+n\s*(==\s*0|<=\s*1|<\s*2)', r_code):
                score += 0.15
            if re.search(r'return\s+n\s*\*\s*factorial\s*\(\s*n\s*-\s*1\s*\)', r_code):
                score += 0.20
            
            # Ejecución segura (0.5 pts)
            result = SecureSandbox.execute(r_code)
            if result['success']:
                namespace = result['namespace']
                if 'factorial' in namespace:
                    try:
                        if callable(namespace['factorial']):
                            computed = namespace['factorial'](n)
                            if computed == solucion:
                                score += 0.50
                    except Exception:
                        pass
            
            return min(score, 1.0)
        
        return Tarea("codigo", prompt, validar, dificultad, "Función recursiva Python")


# --- Entorno Principal ---
class EntornoDificil:
    """
    Entorno de evaluación con tareas de alta presión selectiva.
    Implementa el mecanismo de 'La Reina Roja' para escalar dificultad.
    """
    
    DEFAULT_MAX_TOKENS = 600  # Suficiente para CoT + código
    
    def __init__(self, backend: LLMBackend, max_new_tokens: int = None):
        self.backend = backend
        self.max_new_tokens = max_new_tokens or self.DEFAULT_MAX_TOKENS
        self.fabrica_tareas = TareaFactoryDificil()
        self.fabrica_logica = TareaFactoryLogica()
        
        # Inicializar contador de sílabas para español
        try:
            self.contador_silabas = pyphen.Pyphen(lang='es_ES')
        except Exception:
            self.contador_silabas = pyphen.Pyphen(lang='es')
        
        self.tareas = self._crear_tareas_iniciales()
    
    def _contar_silabas(self, texto: str) -> int:
        palabras = re.findall(r'\w+', texto.lower())
        return sum(len(self.contador_silabas.inserted(p).split('-')) for p in palabras)
    
    def _crear_tareas_iniciales(self) -> List[Tarea]:
        def validar_poesia_dificil(r: str) -> float:
            score = 0.0
            versos = [v.strip() for v in r.split('\n') if v.strip()]
            
            if len(versos) < 3:
                return 0.0
            
            versos_validos = versos[:4] if len(versos) >= 4 else versos
            
            # Puntos por formato básico (0.2)
            if len(versos_validos) >= 4:
                score += 0.2
            elif len(versos_validos) >= 3:
                score += 0.1
            
            # Análisis de Rima ABAB usando validador mejorado (0.4)
            score += RhymeValidator.validate_abab(versos_validos) * 0.4
            
            # Análisis de Métrica - Endecasílabos (0.4)
            conteos = [self._contar_silabas(v) for v in versos_validos]
            for sil in conteos:
                # Tolerancia de ±2 sílabas sobre el objetivo de 11
                if abs(sil - 11) <= 2:
                    score += 0.1
                elif abs(sil - 11) <= 3:
                    score += 0.05
            
            return min(score, 1.0)
        
        tareas_fijas = [
            Tarea(
                "extraccion_penalizada",
                "Párrafo: 'De los planetas rocosos, Marte es el más alejado del Sol. "
                "Sin embargo, el planeta más grande, conocido por su Gran Mancha Roja, "
                "es un gigante gaseoso.' Responde únicamente con el nombre del gigante gaseoso.",
                lambda r: (fuzz.ratio(r.strip().lower(), 'júpiter') / 100.0) / 
                         (2.0 if 'marte' in r.strip().lower() else 1.0),
                dificultad=1,
                descripcion="Extracción NLP con penalización"
            ),
            self.fabrica_logica.crear_silogismo_complejo(dificultad=1),
            Tarea(
                "poesia_metrica",
                "Crea una cuarteta (4 versos) con rima ABAB. "
                "Cada verso debe tener 11 sílabas (Endecasílabos).",
                validar_poesia_dificil,
                dificultad=2,
                descripcion="Poesía con métrica estricta"
            )
        ]
        
        tareas_adaptables = [
            self.fabrica_tareas.crear_tarea_matematica(1),
            self.fabrica_tareas.crear_tarea_codigo(1)
        ]
        
        return tareas_fijas + tareas_adaptables
    
    def adaptar_entorno(self):
        """Mecánica de 'La Reina Roja': escala dificultad si la población mejora."""
        ids_adaptables = ["matematicas", "codigo", "razonamiento_cot_silogismo"]
        
        candidatos = [t for t in self.tareas if any(x in t.id_tarea for x in ids_adaptables)]
        if not candidatos:
            return
        
        tarea_a_mutar = random.choice(candidatos)
        nueva_dificultad = tarea_a_mutar.dificultad + 1
        
        print(f"\n>> ⚠️ ADAPTANDO ENTORNO: Tarea '{tarea_a_mutar.id_tarea}' → dificultad {nueva_dificultad}")
        
        nueva_tarea = None
        if "matematicas" in tarea_a_mutar.id_tarea:
            nueva_tarea = self.fabrica_tareas.crear_tarea_matematica(nueva_dificultad)
        elif "codigo" in tarea_a_mutar.id_tarea:
            nueva_tarea = self.fabrica_tareas.crear_tarea_codigo(nueva_dificultad)
        elif "cot" in tarea_a_mutar.id_tarea:
            nueva_tarea = self.fabrica_logica.crear_silogismo_complejo(nueva_dificultad)
        
        if nueva_tarea:
            self.tareas = [t for t in self.tareas if t.id_tarea != tarea_a_mutar.id_tarea] + [nueva_tarea]
    
    def evaluar_poblacion(self, poblacion: List[OrganismoLLM], num_evaluaciones: int,
                          ponderar_por_dificultad: bool = True):
        """
        Evalúa toda la población usando inferencia por lotes.
        
        Args:
            poblacion: Lista de organismos a evaluar
            num_evaluaciones: Número de evaluaciones por organismo por tarea
            ponderar_por_dificultad: Si True, tareas más difíciles pesan más en fitness
        """
        prompts = []
        organismos_y_tareas = []
        
        # 1. Preparar Prompts
        for org in poblacion:
            # Resetear puntuaciones
            org.puntuaciones_tareas = {}
            org.dificultades_tareas = {}
            
            for _ in range(num_evaluaciones):
                for tarea in self.tareas:
                    prompts.append(org.compilar_prompt(tarea.prompt))
                    organismos_y_tareas.append((org, tarea))
        
        # 2. Inferencia Masiva
        print(f"🧠 Evaluando {len(prompts)} prompts...")
        resultados = self.backend.generate(prompts, max_new_tokens=self.max_new_tokens, batch_size=16)
        
        # 3. Asignación de Puntajes
        for i, resultado_dict in enumerate(resultados):
            organismo, tarea = organismos_y_tareas[i]
            puntuacion = tarea.evaluar_respuesta(resultado_dict['generated_text'])
            organismo.registrar_puntuacion(tarea.id_tarea, puntuacion, tarea.dificultad)
        
        # 4. Cálculo final de Fitness
        for org in poblacion:
            org.calcular_fitness(ponderar_por_dificultad=ponderar_por_dificultad)
    
    def get_info_tareas(self) -> List[Dict[str, Any]]:
        """Retorna información sobre las tareas actuales del entorno."""
        return [
            {
                "id": t.id_tarea,
                "descripcion": t.descripcion,
                "dificultad": t.dificultad
            }
            for t in self.tareas
        ]
