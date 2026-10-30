# juez_llm.py
# Tier 3 de validación: JUEZ LLM para tareas abiertas (síntesis, creatividad,
# pragmática, calidad del razonamiento) donde ni regex ni embeddings bastan.
#
# Principios (para que sea defendible ante reviewer):
#   - Modelo juez FIJO y de OTRA familia que el modelo evolucionado (evita que el
#     modelo se auto-favorezca). Default gemma4:12b (neutro vs qwen/deepseek).
#   - temperature=0 + rúbrica fija -> reproducible.
#   - Devuelve un escalar 0.0-1.0; el juez NO ve la identidad, solo tarea + respuesta.
#
# NOTA: Claude (el asistente) NO es el juez en producción — no está presente durante
# los runs de 1000 generaciones. El juez es este modelo local fijo. Claude diseña las
# rúbricas y los prompts.

import re
import requests

JUEZ_DEFAULT = "gemma4:12b"   # cambiar si se usa otro; debe estar en `ollama list`


def _consultar_juez(prompt: str, modelo: str, host: str, puerto: int, timeout: int) -> str:
    payload = {"model": modelo, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.0, "num_predict": 16}, "think": False}
    r = requests.post(f"http://{host}:{puerto}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")


def _parse_score(texto: str):
    m = re.search(r'(\d+(?:\.\d+)?)', texto)
    if not m:
        return None
    val = float(m.group(1))
    if val > 1.0:            # por si responde 0-100 o 0-10
        val = val / (100.0 if val > 10 else 10.0)
    return max(0.0, min(1.0, val))


def v_juez(rubrica: str, prompt_tarea: str, modelo: str = JUEZ_DEFAULT,
           host: str = "localhost", puerto: int = 11434, timeout: int = 60):
    """
    Devuelve un validador que pide al modelo juez puntuar la respuesta según la rúbrica.
    El validador recibe SOLO la respuesta (firma de Tarea); la tarea y la rúbrica quedan
    capturadas en el closure.
    """
    def f(respuesta: str) -> float:
        judge_prompt = (
            "Eres un evaluador experto y estricto. Puntúa la RESPUESTA del 0.0 al 1.0 "
            "según la RÚBRICA. Responde ÚNICAMENTE con el número decimal, sin explicación.\n\n"
            f"RÚBRICA: {rubrica}\n\n"
            f"TAREA: {prompt_tarea}\n\n"
            f"RESPUESTA: {respuesta}\n\n"
            "PUNTAJE (0.0-1.0):"
        )
        try:
            out = _consultar_juez(judge_prompt, modelo, host, puerto, timeout)
            score = _parse_score(out)
            return score if score is not None else 0.0
        except Exception:
            return 0.0  # si el juez falla, no rompe la evaluación
    return f


def juez_disponible(modelo: str = JUEZ_DEFAULT, host: str = "localhost", puerto: int = 11434) -> bool:
    try:
        tags = requests.get(f"http://{host}:{puerto}/api/tags", timeout=5).json()
        return any(m["name"] == modelo for m in tags.get("models", []))
    except Exception:
        return False


if __name__ == "__main__":
    # Smoke test: requiere el modelo juez en ollama. Correr en terminal.
    import sys
    modelo = sys.argv[1] if len(sys.argv) > 1 else JUEZ_DEFAULT
    print(f"Juez '{modelo}' disponible:", juez_disponible(modelo))
    if juez_disponible(modelo):
        v = v_juez("Premia respuestas concisas y correctas; penaliza divagación.",
                   "¿Cuál es la capital de Francia?", modelo=modelo)
        for resp in ["París", "La capital de Francia es la hermosa ciudad de París, llena de historia y arte sin fin."]:
            print(f"  '{resp[:40]}...' -> {v(resp):.2f}")
