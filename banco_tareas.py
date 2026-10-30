# banco_tareas.py
# Banco de tareas estratificado para EvolutionLLM.
#
# 10 categorías auto-evaluables × 5 niveles (L1 piso → L5 techo) × ~3 instancias.
# Reusa Tarea + validadores del entorno original (CoTValidator, RhymeValidator,
# SecureSandbox). Las categorías 11-12 (pragmática / generación divergente) requieren
# LLM-juez y NO están aquí (se añadirán con un backend juez aparte).
#
# Uso como librería:
#   from banco_tareas import construir_banco, particion_train_heldout
#   banco = construir_banco()
#   train, heldout = particion_train_heldout(banco, frac_heldout=0.4)
#
# Smoke test (sin GPU): evolution/bin/python banco_tareas.py

import re
import json as _json
import random
from typing import List, Callable

import pyphen
from thefuzz import fuzz
from entorno_dificil import Tarea, CoTValidator, RhymeValidator, SecureSandbox, limite_tiempo
from validacion_semantica import v_cobertura  # Tier 2: cobertura semántica (síntesis)

try:
    _PYPHEN = pyphen.Pyphen(lang='es_ES')
except Exception:
    _PYPHEN = pyphen.Pyphen(lang='es')


def _silabas(texto: str) -> int:
    palabras = re.findall(r'\w+', texto.lower())
    return sum(len(_PYPHEN.inserted(p).split('-')) for p in palabras)


def _norm(s: str) -> str:
    return s.strip().lower()


# ============================================================
#  FÁBRICAS DE VALIDADORES
# ============================================================
def v_extraccion(esperado, distractores=()) -> Callable:
    esperado = [esperado] if isinstance(esperado, str) else list(esperado)

    def f(r: str) -> float:
        rl = _norm(r)
        toks = re.findall(r'\w+', rl)
        base = 0.0
        for e in esperado:
            en = _norm(e)
            if en in toks:                       # palabra/numero exacto presente
                base = 1.0
                break
            # fuzzy ALTO contra cada token (tolera acentos/plurales), no parcial laxo
            best_tok = max((fuzz.ratio(en, t) for t in toks), default=0) / 100.0
            if best_tok >= 0.85:
                base = max(base, best_tok)
        # penaliza verbosidad: responder mucho más que la respuesta literal
        if len(toks) > 8:
            base *= 0.7
        # penaliza mencionar un distractor
        if any(_norm(d) in toks for d in distractores):
            base /= 2.0
        return min(base, 1.0)
    return f


def v_math(esperado: float, tol: float = 0.01) -> Callable:
    def f(r: str) -> float:
        rc = r.replace(',', '.')
        for num in re.findall(r'-?\d+\.?\d*', rc):
            try:
                if abs(float(num) - esperado) < tol:
                    return 1.0
            except ValueError:
                continue
        return 0.0
    return f


def v_keyword(esperado, prohibidos=()) -> Callable:
    esperado = [esperado] if isinstance(esperado, str) else list(esperado)

    def f(r: str) -> float:
        rl = _norm(r)
        if not any(_norm(e) in rl for e in esperado):
            return 0.0
        score = 1.0
        if any(_norm(p) in rl for p in prohibidos):
            score *= 0.3
        return score
    return f


def v_codigo(func_name: str, casos: list) -> Callable:
    """casos: lista de (args_tuple, esperado)."""
    def f(r: str) -> float:
        if "```python" in r:
            code = r.split("```python", 1)[1].split("```")[0]
        elif "```" in r:
            code = r.split("```", 1)[1].split("```")[0]
        else:
            code = r
        score = 0.0
        if f'def {func_name}' in code:
            score += 0.3
        res = SecureSandbox.execute(code)
        if res.get('success') and func_name in res.get('namespace', {}):
            fn = res['namespace'][func_name]
            if callable(fn):
                ok = 0
                for args, esperado in casos:
                    try:
                        # timeout por llamada: mata bucles infinitos del código generado
                        with limite_tiempo(2.0):
                            resultado = fn(*args)
                        if resultado == esperado:
                            ok += 1
                    except Exception:
                        pass
                score += 0.7 * (ok / len(casos))
        return min(score, 1.0)
    return f


def v_rima_pareado() -> Callable:
    def f(r: str) -> float:
        versos = [v.strip() for v in r.split('\n') if v.strip()]
        if len(versos) < 2:
            return 0.0
        w = [re.findall(r'\w+', v) for v in versos[:2]]
        if not (w[0] and w[1]):
            return 0.0
        return 1.0 if RhymeValidator.words_rhyme(w[0][-1], w[1][-1]) else 0.3
    return f


def v_cuarteta_abab(target_silabas: int = 11) -> Callable:
    def f(r: str) -> float:
        versos = [v.strip() for v in r.split('\n') if v.strip()]
        if len(versos) < 4:
            return 0.0
        vv = versos[:4]
        score = 0.2  # formato
        score += RhymeValidator.validate_abab(vv) * 0.4
        for sil in (_silabas(v) for v in vv):
            if abs(sil - target_silabas) <= 2:
                score += 0.1
        return min(score, 1.0)
    return f


def v_silabas_exactas(target: int, n_versos: int) -> Callable:
    def f(r: str) -> float:
        versos = [v.strip() for v in r.split('\n') if v.strip()][:n_versos]
        if len(versos) < n_versos:
            return 0.0
        ok = sum(1 for v in versos if abs(_silabas(v) - target) <= 1)
        return ok / n_versos
    return f


def v_lipograma(letra: str, min_palabras: int = 5) -> Callable:
    def f(r: str) -> float:
        cuerpo = r.strip()
        if len(re.findall(r'\w+', cuerpo)) < min_palabras:
            return 0.0
        return 1.0 if letra.lower() not in cuerpo.lower() else 0.0
    return f


def v_sintesis(max_palabras: int, keywords) -> Callable:
    keywords = list(keywords)

    def f(r: str) -> float:
        rl = _norm(r)
        n = len(re.findall(r'\w+', r))
        cov = sum(1 for k in keywords if _norm(k) in rl) / len(keywords) if keywords else 1.0
        # penaliza exceder el límite de palabras (castiga verborrea)
        pen = 1.0 if n <= max_palabras else max(0.0, 1.0 - (n - max_palabras) / max_palabras)
        return cov * pen
    return f


def v_si_no(esperado: str) -> Callable:
    esperado = _norm(esperado)

    def f(r: str) -> float:
        toks = re.findall(r'\b(s[ií]|no)\b', _norm(r))
        if not toks:
            return 0.0
        ult = toks[-1].replace('si', 'sí')
        return 1.0 if ult == esperado.replace('si', 'sí') else 0.0
    return f


def v_json_campo(campo: str, esperado) -> Callable:
    def f(r: str) -> float:
        m = re.search(r'\{.*\}', r, re.DOTALL)
        if not m:
            return 0.0
        try:
            obj = _json.loads(m.group(0))
        except Exception:
            return 0.2  # intentó pero JSON inválido
        return 1.0 if str(obj.get(campo)) == str(esperado) else 0.3
    return f


def v_json_campos(esperados: dict) -> Callable:
    """JSON con múltiples campos: fracción de campos correctos."""
    def f(r: str) -> float:
        m = re.search(r'\{.*\}', r, re.DOTALL)
        if not m:
            return 0.0
        try:
            obj = _json.loads(m.group(0))
        except Exception:
            return 0.2
        ok = sum(1 for k, v in esperados.items() if str(obj.get(k)) == str(v))
        return ok / len(esperados)
    return f


def v_mayusculas(palabra: str) -> Callable:
    """La respuesta debe contener la palabra en MAYÚSCULAS exactas."""
    def f(r: str) -> float:
        return 1.0 if palabra.upper() in r else 0.0
    return f


def v_lista_n(n: int) -> Callable:
    def f(r: str) -> float:
        items = re.findall(r'^\s*(?:\d+[.)]|[-•])\s+.+', r, re.MULTILINE)
        if not items:
            return 0.0
        return 1.0 if len(items) == n else max(0.0, 1.0 - abs(len(items) - n) * 0.4)
    return f


def v_sin_puntuacion() -> Callable:
    def f(r: str) -> float:
        cuerpo = r.strip()
        if len(re.findall(r'\w+', cuerpo)) < 3:
            return 0.0
        return 1.0 if not re.search(r'[.,;:!?¡¿]', cuerpo) else 0.0
    return f


def v_csv(ncols: int, nfilas: int) -> Callable:
    def f(r: str) -> float:
        lineas = [l for l in r.strip().splitlines() if ',' in l]
        if not lineas:
            return 0.0
        filas_ok = sum(1 for l in lineas if len(l.split(',')) == ncols)
        score = 0.5 * (filas_ok / max(len(lineas), 1))
        score += 0.5 if filas_ok >= nfilas else 0.0
        return min(score, 1.0)
    return f


def v_cot(esperado: str) -> Callable:
    def f(r: str) -> float:
        if CoTValidator.detect_marker_stuffing(r):
            return 0.1
        return CoTValidator.validate(r, expected_conclusion=esperado)
    return f


def v_rechazo() -> Callable:
    """Premisa falsa: debe negar / no dar un dato inventado."""
    NEG = ["no ocurrió", "no ha ocurrido", "no existe", "no es cierto", "nunca",
           "no sucedió", "falso", "no hay registro", "no ha sucedido", "no se ha"]

    def f(r: str) -> float:
        rl = _norm(r)
        nego = any(n in rl for n in NEG)
        da_anio = bool(re.search(r'\b(1[5-9]\d\d|20\d\d|21\d\d)\b', rl))
        if nego and not da_anio:
            return 1.0
        if nego and da_anio:
            return 0.5
        return 0.0
    return f


# ============================================================
#  CONSTRUCTOR DE TAREAS
# ============================================================
def _T(cat: str, nivel: int, i: int, prompt: str, validador: Callable) -> Tarea:
    return Tarea(f"{cat}_L{nivel}_{i}", prompt, validador, dificultad=nivel, descripcion=cat)


def construir_banco() -> List[Tarea]:
    T: List[Tarea] = []

    # ---- 1. EXTRACCIÓN LITERAL 🧬 ----
    c = "extraccion"
    T += [
        _T(c,1,1,"¿Cuál es la capital de Francia? Responde solo con el nombre.", v_extraccion("parís",("francia",))),
        _T(c,1,2,"¿Cuál es la capital de Japón? Responde solo con el nombre.", v_extraccion("tokio")),
        _T(c,1,3,"¿Cuál es la capital de Italia? Responde solo con el nombre.", v_extraccion("roma")),
        _T(c,2,1,"Párrafo: 'Marte es el planeta rojo, pero el más grande, con su Gran Mancha Roja, es un gigante gaseoso.' Responde únicamente con el nombre del gigante gaseoso.", v_extraccion("júpiter",("marte",))),
        _T(c,2,2,"El perro de Ana es negro; el de Luis es blanco. ¿De qué color es el perro de Luis? Una palabra.", v_extraccion("blanco",("negro",))),
        _T(c,2,3,"En la caja roja hay 3 lápices y en la azul 5. ¿Cuántos lápices hay en la caja azul? Solo el número.", v_extraccion("5",("3",))),
        _T(c,3,1,"Compré 4 peras, 6 uvas, 9 nueces y 7 ciruelas. ¿Qué cantidad es un número primo? Solo el número.", v_extraccion("7")),
        _T(c,3,2,"Edades: 8, 10, 11, 12. ¿Cuál es prima? Solo el número.", v_extraccion("11")),
        _T(c,3,3,"Pesos: 15, 21, 23, 27. ¿Cuál es primo? Solo el número.", v_extraccion("23")),
        _T(c,4,1,"El director, que antes fue rival de su socio Luis, firmó el contrato. ¿Quién firmó? Responde con el rol.", v_extraccion("director",("luis",))),
        _T(c,4,2,"María le dio el libro a su hermana, quien lo leyó esa noche. ¿Quién leyó el libro? Una palabra.", v_extraccion("hermana",("maría",))),
        _T(c,4,3,"El gato perseguía al ratón hasta que este escapó por el hueco. ¿Quién escapó? Una palabra.", v_extraccion("ratón",("gato",))),
        _T(c,5,1,"No ganó Ana ni quien estaba a su derecha, sino el de su izquierda. Orden: Beto, Ana, Caro. ¿Quién ganó? Un nombre.", v_extraccion("beto",("ana","caro"))),
        _T(c,5,2,"El premio no fue para el más alto ni el más bajo, sino el del medio. Alturas: Sol 1.80, Tom 1.60, Eva 1.70. ¿Quién ganó?", v_extraccion("eva",("sol","tom"))),
        _T(c,5,3,"El evento no fue el lunes ni el miércoles, sino el día entre ambos. ¿Qué día fue? Una palabra.", v_extraccion("martes",("lunes","miércoles"))),
    ]

    # ---- 2. LÓGICA DEDUCTIVA 🧬 ----
    c = "logica"
    T += [
        # capacidad = conclusión correcta (v_si_no); la ESTRUCTURA del razonamiento la
        # capta el módulo de estilo aparte -> sin el confound de CoTValidator.
        _T(c,1,1,"Si llueve, la calle se moja. Llueve. ¿Se moja la calle? Razona paso a paso y termina con sí o no.", v_si_no("sí")),
        _T(c,1,2,"Todos los perros son mamíferos. Rex es un perro. ¿Rex es mamífero? Razona y concluye sí o no.", v_si_no("sí")),
        _T(c,1,3,"Si estudias, apruebas. Estudiaste. ¿Apruebas? Razona paso a paso, sí o no.", v_si_no("sí")),
        _T(c,2,1,"Todos los Gorks son Morks. Ningún Mork es Zork. ¿Puede un Gork ser Zork? Razona y concluye sí o no.", v_si_no("no")),
        _T(c,2,2,"Todos los Flims son Blops. Ningún Blop es Trull. ¿Puede un Flim ser Trull? Razona, sí o no.", v_si_no("no")),
        _T(c,2,3,"Ningún Pim es Pam. Todos los Pom son Pim. ¿Algún Pom es Pam? Razona, sí o no.", v_si_no("no")),
        _T(c,3,1,"Algunos Aros son Beros. Todos los Beros son Ceros. ¿Hay Aros que sean Ceros? Razona, sí o no.", v_si_no("sí")),
        _T(c,3,2,"Todos los X son Y. Algunos Y son Z. Todos los Z son W. ¿Necesariamente algún X es W? Razona, sí o no.", v_si_no("no")),
        _T(c,4,1,"Ana, Beto y Caro están en fila. Ana no es primera. Caro está justo detrás de Ana. ¿Quién es primero? Una palabra.", v_keyword("beto",("ana","caro"))),
        _T(c,4,2,"Cuatro corredores: P llegó antes que Q, Q antes que R, R antes que S. ¿Quién llegó último? Una letra.", v_keyword("s")),
        _T(c,5,1,"A dice: 'B miente'. B dice: 'A y yo decimos la verdad'. Solo uno miente. ¿Quién miente? Una letra.", v_keyword("b",("a",))),
        _T(c,5,2,"En una isla, los caballeros siempre dicen verdad y los bribones siempre mienten. X dice 'soy un bribón'. ¿Es eso posible? Razona, sí o no.", v_si_no("no")),
    ]

    # ---- 3. MATEMÁTICA ----
    c = "math"
    T += [
        _T(c,1,1,"¿Cuánto es 17 + 25? Solo el número.", v_math(42)),
        _T(c,1,2,"¿Cuánto es 8 × 7? Solo el número.", v_math(56)),
        _T(c,1,3,"¿Cuánto es 100 − 37? Solo el número.", v_math(63)),
        _T(c,2,1,"Calcula x en 2·(x + 3) = 10·1. Muestra los pasos. Solo el valor final.", v_math(2)),
        _T(c,2,2,"Calcula x en 3·(x + 4) = 9·5. Muestra los pasos.", v_math(11)),
        _T(c,2,3,"Calcula x en 4·(x + 5) = 8·9. Muestra los pasos.", v_math(13)),
        _T(c,3,1,"Un precio sube 10% y luego baja 10%; quedó en 99. ¿Cuánto era el precio original? Solo el número.", v_math(100)),
        _T(c,3,2,"Resuelve el sistema: x + y = 10, x − y = 4. ¿Cuánto vale x? Solo el número.", v_math(7)),
        _T(c,3,3,"El 30% de un número es 45. ¿Cuál es el número? Solo el número.", v_math(150)),
        _T(c,4,1,"Dos trenes separados 300 km avanzan uno hacia el otro a 60 y 90 km/h. ¿En cuántas horas se cruzan? Solo el número.", v_math(2)),
        _T(c,4,2,"Un grifo llena un tanque en 6 h y otro en 3 h. Juntos, ¿en cuántas horas lo llenan? Solo el número.", v_math(2)),
        _T(c,5,1,"¿De cuántas formas distintas se pueden ordenar 4 libros diferentes en un estante? Solo el número.", v_math(24)),
        _T(c,5,2,"Lanzas dos dados. ¿Cuántas combinaciones (ordenadas) suman 7? Solo el número.", v_math(6)),
    ]

    # ---- 4. PROGRAMACIÓN ----
    c = "codigo"
    T += [
        _T(c,1,1,"Escribe en Python una función `suma(a, b)` que retorne la suma de a y b. Usa bloque ```python.", v_codigo("suma",[((2,3),5),((10,-4),6)])),
        _T(c,1,2,"Escribe `doble(n)` que retorne n multiplicado por 2. Bloque ```python.", v_codigo("doble",[((5,),10),((0,),0)])),
        _T(c,2,1,"Escribe una función recursiva `factorial(n)`. Demuéstrala con factorial de 4. Bloque ```python.", v_codigo("factorial",[((4,),24),((0,),1)])),
        _T(c,2,2,"Escribe `maximo(lista)` que retorne el mayor elemento. Bloque ```python.", v_codigo("maximo",[(([3,7,2],),7),(([-1,-5],),-1)])),
        _T(c,3,1,"Escribe `es_palindromo(s)` que retorne True si s es palíndromo (ignora mayúsculas). Bloque ```python.", v_codigo("es_palindromo",[(("Anita",),False),(("oso",),True),(("",),True)])),
        _T(c,3,2,"Escribe `cuenta_vocales(s)` que retorne cuántas vocales hay. Bloque ```python.", v_codigo("cuenta_vocales",[(("hola",),2),(("xyz",),0)])),
        _T(c,4,1,"Escribe `fizzbuzz(n)` que retorne 'Fizz' si divisible por 3, 'Buzz' por 5, 'FizzBuzz' por ambos, o str(n). Bloque ```python.", v_codigo("fizzbuzz",[((3,),"Fizz"),((5,),"Buzz"),((15,),"FizzBuzz"),((7,),"7")])),
        _T(c,4,2,"Escribe `suma_pares(lista)` que sume solo los números pares. Bloque ```python.", v_codigo("suma_pares",[(([1,2,3,4],),6),(([1,3],),0)])),
        _T(c,5,1,"Escribe `balanceado(s)` que retorne True si los paréntesis están balanceados. Bloque ```python.", v_codigo("balanceado",[(("(())",),True),(("(()",),False),(("",),True)])),
        _T(c,5,2,"Escribe `fib(n)` que retorne el n-ésimo de Fibonacci (fib(0)=0, fib(1)=1). Bloque ```python.", v_codigo("fib",[((0,),0),((1,),1),((7,),13)])),
        # ---- código DIFÍCIL (nivel PhD): algorítmica real, da headroom para deepseek ----
        _T(c,5,3,"Escribe `lcs(a, b)` que retorne la longitud de la subsecuencia común más larga de dos strings (programación dinámica). Bloque ```python.", v_codigo("lcs",[(("ABCBDAB","BDCAB"),4),(("","x"),0),(("abc","abc"),3)])),
        _T(c,5,4,"Escribe `edit_distance(a, b)` (distancia de Levenshtein). Bloque ```python.", v_codigo("edit_distance",[(("kitten","sitting"),3),(("","abc"),3),(("ab","ab"),0)])),
        _T(c,5,5,"Escribe `n_reinas(n)` que retorne el número de soluciones del problema de las N reinas. Bloque ```python.", v_codigo("n_reinas",[((4,),2),((1,),1),((8,),92)])),
        _T(c,5,6,"Escribe `dijkstra(grafo, inicio, fin)` donde grafo es dict {nodo: [(vecino, peso)]}, que retorne el costo mínimo. Bloque ```python.", v_codigo("dijkstra",[(({0:[(1,4),(2,1)],1:[(3,1)],2:[(1,2),(3,5)],3:[]},0,3),4)])),
        _T(c,5,7,"Escribe `max_subarray(nums)` que retorne la suma máxima de un subarreglo contiguo (Kadane). Bloque ```python.", v_codigo("max_subarray",[(([-2,1,-3,4,-1,2,1,-5,4],),6),(([-1,-2],),-1)])),
    ]

    # ---- 5. CREATIVIDAD FORMAL 🧬 ----
    c = "creatividad"
    T += [
        _T(c,1,1,"Escribe UNA sola frase que termine en una palabra que rime con 'canción'.", lambda r: (lambda w: 1.0 if w and RhymeValidator.words_rhyme(w[-1],"canción") else 0.2)(re.findall(r'\w+', r.split(chr(10))[0]) if r.strip() else [])),
        _T(c,1,2,"Escribe UNA frase que termine en una palabra que rime con 'amor'.", lambda r: (lambda w: 1.0 if w and RhymeValidator.words_rhyme(w[-1],"amor") else 0.2)(re.findall(r'\w+', r.split(chr(10))[0]) if r.strip() else [])),
        _T(c,2,1,"Escribe un pareado: dos versos que rimen entre sí (rima AA).", v_rima_pareado()),
        _T(c,2,2,"Escribe dos versos sobre el mar que rimen entre sí.", v_rima_pareado()),
        _T(c,3,1,"Crea una cuarteta (4 versos) con rima ABAB. Cada verso de 11 sílabas (endecasílabos).", v_cuarteta_abab()),
        _T(c,3,2,"Crea una cuarteta ABAB sobre el tiempo, versos de 11 sílabas.", v_cuarteta_abab()),
        _T(c,4,1,"Escribe un haiku de 3 versos con 5, 7 y 5 sílabas respectivamente.", v_silabas_exactas(5,1)),  # aprox: primer verso 5
        _T(c,4,2,"Escribe un verso de exactamente 7 sílabas sobre la lluvia.", v_silabas_exactas(7,1)),
        _T(c,5,1,"Escribe 3 frases coherentes SIN usar nunca la letra 'a'.", v_lipograma("a")),
        _T(c,5,2,"Escribe 2 frases SIN usar nunca la letra 'e'.", v_lipograma("e")),
    ]

    # ---- 6. SÍNTESIS / CONCISIÓN 🧬 ----
    c = "sintesis"
    T += [
        # cobertura SEMÁNTICA de las ideas clave (embeddings) + penalización por verbosidad
        _T(c,1,1,"Resume en UNA sola frase: 'El sol salió. Los pájaros cantaron. La ciudad despertó lentamente.'", v_cobertura(["la ciudad despertó al amanecer"], max_palabras=15)),
        _T(c,1,2,"Resume en UNA frase: 'Llovió toda la noche. El río creció. Las calles se inundaron.'", v_cobertura(["la lluvia inundó las calles"], max_palabras=15)),
        _T(c,2,1,"Resume en ≤10 palabras: 'El cambio en el clima obliga a buscar nuevas fuentes de energía renovable para el futuro.'", v_cobertura(["el clima exige energía renovable"], max_palabras=10)),
        _T(c,2,2,"Resume en ≤10 palabras: 'La empresa analiza datos masivos para tomar cada decisión estratégica con menos riesgo.'", v_cobertura(["usar datos para decidir con menos riesgo"], max_palabras=10)),
        _T(c,3,1,"TL;DR en ≤15 palabras: 'Un nuevo algoritmo de compresión reduce el tamaño de los archivos sin perder calidad, acelerando las descargas.'", v_cobertura(["algoritmo comprime archivos sin perder calidad"], max_palabras=15)),
        _T(c,3,2,"TL;DR en ≤15 palabras: 'La vacuna entrena al sistema inmune para reconocer al virus antes de que cause enfermedad grave.'", v_cobertura(["la vacuna prepara al sistema inmune contra el virus"], max_palabras=15)),
        _T(c,4,1,"Resume en ≤20 palabras conservando los números: 'El proyecto costó 5 millones, duró 3 años e involucró a 200 personas de 4 países.'", v_sintesis(20,["5","3","200","4"])),
        _T(c,5,1,"Comprime en ≤18 palabras conservando TODAS las cifras: 'La maratón de 42 km tuvo 8000 corredores, 12 puestos de agua y 3 horas de tiempo límite.'", v_sintesis(18,["42","8000","12","3"])),
    ]

    # ---- 7. FORMATO ESTRICTO 🧬 ----
    c = "formato"
    T += [
        _T(c,1,1,"Responde únicamente SÍ o NO: ¿el Sol es una estrella?", v_si_no("sí")),
        _T(c,1,2,"Responde únicamente SÍ o NO: ¿los peces vuelan?", v_si_no("no")),
        _T(c,2,1,"Responde SOLO en JSON válido con la forma {\"respuesta\": <numero>} para 3 × 4.", v_json_campo("respuesta",12)),
        _T(c,2,2,"Responde SOLO en JSON válido {\"respuesta\": <numero>} para 20 − 5.", v_json_campo("respuesta",15)),
        _T(c,3,1,"Lista numerada de EXACTAMENTE 3 colores primarios. Solo la lista.", v_lista_n(3)),
        _T(c,3,2,"Lista numerada de EXACTAMENTE 3 planetas del sistema solar. Solo la lista.", v_lista_n(3)),
        _T(c,4,1,"Responde la frase 'el cielo es azul hoy' SIN usar ningún signo de puntuación.", v_sin_puntuacion()),
        _T(c,4,2,"Describe un árbol en una frase SIN usar ningún signo de puntuación.", v_sin_puntuacion()),
        _T(c,5,1,"Devuelve una tabla CSV con columnas nombre,edad,ciudad y 2 filas de ejemplo. Solo el CSV.", v_csv(3,2)),
        _T(c,5,2,"Devuelve un CSV de 3 columnas (producto,precio,stock) con 2 filas. Solo el CSV.", v_csv(3,2)),
    ]

    # ---- 8. ABSTRACTO / ANALÓGICO ----
    c = "abstracto"
    T += [
        _T(c,1,1,"Completa la secuencia: 2, 4, 6, ? Solo el número.", v_math(8)),
        _T(c,1,2,"Completa: 5, 10, 15, ? Solo el número.", v_math(20)),
        _T(c,2,1,"Analogía: guante es a mano como zapato es a ? Una palabra.", v_keyword(["pie","pies"])),
        _T(c,2,2,"Analogía: día es a noche como verano es a ? Una palabra.", v_keyword("invierno")),
        _T(c,3,1,"Secuencia: 1, 2, 4, 7, 11, ? (se suma 1, 2, 3, 4...). Solo el número.", v_math(16)),
        _T(c,3,2,"Secuencia: 1, 4, 9, 16, ? (cuadrados). Solo el número.", v_math(25)),
        _T(c,4,1,"¿Cuál sobra y por qué: perro, gato, roble, caballo? Indica la palabra que sobra.", v_keyword("roble")),
        _T(c,4,2,"¿Cuál sobra: rosa, tulipán, zanahoria, margarita? Indica la palabra.", v_keyword("zanahoria")),
        _T(c,5,1,"Regla: 'gato'→'g', 'perro'→'p'. Aplica la misma regla a 'tigre'. Solo el resultado.", v_keyword("t")),
        _T(c,5,2,"Si 2→6, 3→9, 4→12 (multiplica por 3), ¿a qué va 7? Solo el número.", v_math(21)),
    ]

    # ---- 9. FACTUAL CLOSED-BOOK 🧬(L4) ----
    c = "factual"
    T += [
        _T(c,1,1,"¿Cuántos lados tiene un triángulo? Solo el número.", v_math(3)),
        _T(c,1,2,"¿De qué color es el cielo despejado de día? Una palabra.", v_keyword("azul")),
        _T(c,2,1,"¿Quién escribió 'Don Quijote de la Mancha'? Solo el apellido.", v_keyword("cervantes")),
        _T(c,2,2,"¿En qué continente está Egipto? Una palabra.", v_keyword(["áfrica","africa"])),
        _T(c,3,1,"¿Cuál es el símbolo químico del oro? Solo el símbolo.", v_keyword(["au","oro","gold"])),
        _T(c,3,2,"¿Cuál es el planeta más cercano al Sol? Una palabra.", v_keyword(["mercurio","mercury"])),
        _T(c,4,1,"¿En qué año la NASA colonizó Marte con humanos?", v_rechazo()),
        _T(c,4,2,"¿Cómo se llama el océano que separa España de Brasil por tierra firme?", v_rechazo()),
        _T(c,5,1,"¿En qué año publicaron Cooley y Tukey el algoritmo de la transformada rápida de Fourier (FFT)? Solo el año.", v_math(1965)),
        _T(c,5,2,"¿En qué año se fundó la teoría de la información de Shannon (paper 'A Mathematical Theory of Communication')? Solo el año.", v_math(1948)),
    ]

    # ---- 10. CAUSAL / CONTRAFACTUAL ----
    c = "causal"
    T += [
        _T(c,1,1,"Si sueltas una pelota en el aire, ¿sube o baja? Una palabra.", v_keyword(["baja","cae","desciende","abajo","caer","descender"],("sube","asciende"))),
        _T(c,1,2,"Si tocas el fuego, ¿te quemas o te enfrías? Una palabra.", v_keyword(["quema","quemas","quemar","arde"],("enfría","enfrías"))),
        _T(c,2,1,"Ordena por causa: 'el suelo está mojado', 'llovió'. ¿Qué ocurrió PRIMERO? Una palabra clave.", v_keyword(["llovió","lluvia","llover"])),
        _T(c,2,2,"Eventos: 'el vaso se rompió', 'el vaso cayó'. ¿Cuál fue la causa? Una palabra clave.", v_keyword(["cayó","caída","caer","caída del vaso"])),
        _T(c,3,1,"Contrafactual: si NO hubiera llovido, ¿el suelo estaría mojado por la lluvia? Razona y concluye sí o no.", v_si_no("no")),
        _T(c,3,2,"Contrafactual: si la semilla no recibe agua, ¿germinará? Razona y concluye sí o no.", v_si_no("no")),
        _T(c,4,1,"Cadena: el viento tira el árbol, el árbol corta el cable, el cable corta la luz. Si hay viento fuerte, ¿qué pasa con la luz? Una palabra.", v_keyword(["corta","corte","apaga","sin"])),
        _T(c,5,1,"En verano suben las ventas de helados y los ahogamientos. ¿Los helados causan ahogamientos, o hay una causa común? Responde 'causa común' o 'helados'.", v_keyword(["común","calor","verano","temperatura"],("helados causan",))),
    ]

    # ============================================================
    #  CAPA DE MÁXIMA DIFICULTAD (L6-L7) — headroom para deepseek
    #  qwen3:1.7b mayormente las fallará (= piso); discriminarán en deepseek.
    # ============================================================
    T += [
        # --- matemática experta (numérico verificable) ---
        _T("math",6,1,"¿Cuántos números primos hay entre 1 y 100? Solo el número.", v_math(25)),
        _T("math",6,2,"¿Cuál es el mínimo común múltiplo de los números del 1 al 10? Solo el número.", v_math(2520)),
        _T("math",7,1,"¿Cuál es el último dígito de 7 elevado a 100? Solo el dígito.", v_math(1)),
        # --- conteo lógico (inclusión-exclusión) ---
        _T("logica",6,1,"En un grupo todos hablan inglés o francés. 20 hablan inglés, 15 francés y 5 ambos. ¿Cuántas personas hay en total? Solo el número.", v_math(30)),
        _T("logica",6,2,"Todos los Xat son Yor. Ningún Yor es Zif. ¿Puede algún Xat ser Zif? Razona y concluye sí o no.", v_si_no("no")),
        # --- código nivel experto (DP) ---
        _T("codigo",6,1,"Escribe `min_monedas(monedas, objetivo)` que retorne el mínimo número de monedas para sumar 'objetivo', o -1 si es imposible. Bloque ```python.", v_codigo("min_monedas",[(([1,2,5],11),3),(([2],3),-1),(([1],0),0)])),
        _T("codigo",7,1,"Escribe `knapsack(pesos, valores, W)` (mochila 0/1) que retorne el valor máximo sin exceder capacidad W. Bloque ```python.", v_codigo("knapsack",[(([1,3,4],[15,20,30],4),35),(([5],[10],4),0)])),
        # --- abstracto experto ---
        _T("abstracto",6,1,"Completa la secuencia: 2, 3, 5, 7, 11, 13, ? Solo el número.", v_math(17)),
        _T("abstracto",7,1,"Secuencia 'look-and-say': 1, 11, 21, 1211, 111221, ? Responde solo el siguiente término.", v_keyword("312211")),
        # --- factual cola larga ---
        _T("factual",6,1,"¿Cuál es el número atómico del oro? Solo el número.", v_math(79)),
        _T("factual",6,2,"¿Cuántos huesos tiene el cuerpo humano adulto? Solo el número.", v_math(206)),
        # --- causal con confusor (Simpson-like) ---
        _T("causal",6,1,"Los niños con pies más grandes leen mejor. ¿Tener pies grandes causa leer mejor, o hay un factor común? Responde 'factor común' o 'pies'.", v_keyword(["común","edad","factor","crecer","mayores"],("pies causan",))),
        # --- formato estricto experto (JSON multi-campo) ---
        _T("formato",6,1,"Devuelve SOLO un JSON válido con la forma {\"suma\": <n>, \"producto\": <n>} para los números 3 y 4.", v_json_campos({"suma":7,"producto":12})),
        # --- síntesis dura: comprimir conservando 3 ideas distintas (multi-keypoint + corto) ---
        _T("sintesis",6,1,"Resume en ≤25 palabras conservando las TRES ideas: 'La empresa creció 40% este año. Contrató a 200 empleados nuevos. Pero su deuda se duplicó.'",
           v_cobertura(["la empresa creció","contrató empleados nuevos","la deuda se duplicó"], umbral=0.55, max_palabras=25)),
        _T("sintesis",7,1,"Comprime en ≤20 palabras conservando las TRES ideas: 'El plan reduce impuestos a las pymes. Aumenta la inversión en salud. Y recorta el gasto militar.'",
           v_cobertura(["reduce impuestos a pymes","aumenta inversión en salud","recorta gasto militar"], umbral=0.55, max_palabras=20)),
        # --- extracción con triple distractor ---
        _T("extraccion",6,1,"En la carrera: Ana llegó tras Beto, Caro antes que Ana, y Dani el último. Beto no fue primero. ¿Quién llegó primero? Un nombre.", v_extraccion("caro",("ana","beto","dani"))),
    ]

    # ============================================================
    #  CAPA MEDIA (L8) — engrosa bins 2-5 (zona que discrimina capacidad en qwen3).
    #  Más difícil que lo trivial, más fácil que la capa experta. L8 = etiqueta única.
    # ============================================================
    T += [
        _T("math",8,1,"Un triángulo tiene ángulos de 40° y 75°. ¿Cuánto mide el tercer ángulo? Solo el número.", v_math(65)),
        _T("math",8,2,"¿Cuál es la suma de los números del 1 al 10? Solo el número.", v_math(55)),
        _T("logica",8,1,"Si A es mayor que B y B es mayor que C, ¿es A mayor que C? Razona y concluye sí o no.", v_si_no("sí")),
        _T("codigo",8,1,"Escribe `es_primo(n)` que retorne True si n es primo. Bloque ```python.", v_codigo("es_primo",[((7,),True),((8,),False),((1,),False),((2,),True)])),
        _T("codigo",8,2,"Escribe `invierte(s)` que retorne el string invertido. Bloque ```python.", v_codigo("invierte",[(("abc",),"cba"),(("",),"")])),
        _T("abstracto",8,1,"¿Qué número sigue: 3, 6, 12, 24, ? Solo el número.", v_math(48)),
        _T("factual",8,1,"¿Cuántos continentes hay en la Tierra? Solo el número.", v_math(7)),
        _T("factual",8,2,"¿Cuál es la capital de Australia? Una palabra.", v_keyword(["canberra"],("sídney","sydney","melbourne"))),
        _T("causal",8,1,"El gallo canta antes del amanecer. ¿El canto del gallo causa el amanecer? Razona y concluye sí o no.", v_si_no("no")),
        _T("extraccion",8,1,"El libro rojo está sobre el azul, y el verde está bajo el azul. ¿Qué color queda en medio? Un color.", v_extraccion("azul",("rojo","verde"))),
        _T("formato",8,1,"Responde la palabra 'hola' en MAYÚSCULAS y nada más.", v_mayusculas("hola")),
        _T("creatividad",8,1,"Escribe 4 líneas donde las iniciales de cada línea formen la palabra AMOR (acróstico).",
           lambda r: (lambda L: 1.0 if len(L) >= 4 and "".join(x.strip()[0].upper() for x in L[:4] if x.strip()) == "AMOR" else 0.0)([x for x in r.split(chr(10)) if x.strip()])),
    ]

    return T


# ============================================================
#  TAREAS ABIERTAS (Tier 3: juez LLM) — SOLO para el termómetro, NO para selección
# ============================================================
def tareas_abiertas(modelo_juez: str = None) -> List[Tarea]:
    """
    Tareas de calidad abierta evaluadas por juez LLM (creatividad genuina, pragmática).
    Se añaden SOLO al termómetro de medición (costo de juez por evaluación), nunca a la
    presión selectiva. 🧬 categorías donde AXIS contemplativo podría DESTACAR (no solo perder).
    """
    from juez_llm import v_juez, JUEZ_DEFAULT
    mj = modelo_juez or JUEZ_DEFAULT
    T: List[Tarea] = []

    c = "pragmatica"  # 🧬 juicio/empatía contextual
    items = [
        (2, "Un amigo está muy triste porque hoy perdió su trabajo. ¿Qué le dirías?",
         "Premia empatía genuina Y un apoyo o paso útil concreto. Penaliza frialdad, clichés vacíos o divagar sin sustancia."),
        (3, "Tu equipo entregó tarde un proyecto importante. Como líder, ¿qué dirías en la reunión?",
         "Premia responsabilidad, tono constructivo y una acción concreta. Penaliza culpar a otros o vaguedad."),
        (4, "Un niño pregunta por qué el cielo es azul. Explícaselo de forma simple y correcta.",
         "Premia precisión científica adaptada a un niño (dispersión de la luz). Penaliza error factual o jerga incomprensible."),
    ]
    for niv, prompt, rub in items:
        T.append(_T(c, niv, 1, prompt, v_juez(rub, prompt, modelo=mj)))

    c = "creatividad_abierta"  # 🧬 generación divergente / metáfora
    items = [
        (2, "Escribe una metáfora original sobre el paso del tiempo. Una sola frase.",
         "Premia originalidad e imagen evocadora coherente. Penaliza clichés ('el tiempo vuela') o incoherencia."),
        (3, "Inventa tres usos no obvios para un ladrillo.",
         "Premia creatividad y variedad real (no usos triviales de construcción). Penaliza repetición o usos obvios."),
        (4, "Escribe el primer párrafo de un cuento que empiece con una contradicción intrigante.",
         "Premia gancho narrativo, contradicción genuina y prosa fluida. Penaliza apertura plana o incoherente."),
    ]
    for niv, prompt, rub in items:
        T.append(_T(c, niv, 1, prompt, v_juez(rub, prompt, modelo=mj)))

    return T


# ============================================================
#  PARTICIÓN train / held-out (estratificada por categoría×nivel)
# ============================================================
def particion_train_heldout(banco: List[Tarea], frac_heldout: float = 0.4, seed: int = 1234):
    rng = random.Random(seed)
    grupos = {}
    for t in banco:
        m = re.match(r'(.+)_L(\d+)_\d+', t.id_tarea)
        clave = (m.group(1), m.group(2)) if m else (t.id_tarea, "0")
        grupos.setdefault(clave, []).append(t)

    train, heldout = [], []
    for clave, items in grupos.items():
        items = items[:]
        rng.shuffle(items)
        n_held = max(1, round(len(items) * frac_heldout)) if len(items) > 1 else 0
        heldout += items[:n_held]
        train += items[n_held:]
    return train, heldout


if __name__ == "__main__":
    banco = construir_banco()
    cats = {}
    for t in banco:
        cats.setdefault(t.descripcion, set()).add(t.dificultad)
    print(f"Banco construido: {len(banco)} tareas, {len(cats)} categorías\n")
    from collections import Counter
    porcat = Counter(t.descripcion for t in banco)
    for c in sorted(porcat):
        niveles = sorted({t.dificultad for t in banco if t.descripcion == c})
        print(f"  {c:14s}: {porcat[c]:2d} tareas | niveles {niveles}")
    train, heldout = particion_train_heldout(banco)
    print(f"\nPartición: train={len(train)} | held-out={len(heldout)}")
    # validar que cada categoría×nivel aparece en held-out
    held_claves = {re.match(r'(.+)_L(\d+)_', t.id_tarea).groups() for t in heldout}
    print(f"Categoría×nivel cubiertos en held-out: {len(held_claves)}")
