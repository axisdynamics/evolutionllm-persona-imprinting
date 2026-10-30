#!/usr/bin/env python3
"""
Construye este repositorio público a partir del repositorio privado de trabajo.

Nada se transforma: las especificaciones de persona, los genomas y las mediciones van
tal como se escribieron durante los experimentos. El único criterio es de SELECCIÓN —
qué entra y qué no— y está declarado abajo en listas explícitas, para que la frontera
del release sea legible y no dependa de un `.gitignore`.

El layout replica el del repositorio privado a propósito: los scripts de análisis
traen rutas relativas fijas (`results/MIA/...`, `axis_qwen_3.1.txt`), así que corren
aquí sin editar una línea. Quien reproduce ejecuta exactamente lo que ejecutamos.

Uso:
    python3 armar_repo.py --origen /ruta/al/repo/privado [--destino .]
"""
import argparse
import shutil
from pathlib import Path

# Las tres especificaciones de persona. Son modelfiles completos de Ollama: traen su
# `FROM`, su bloque `SYSTEM` (que es el objeto de estudio) y sus `PARAMETER`.
SPECS = ["axis_qwen_3.1.txt", "nexusvex_deep_q3.txt", "biovex_deep_q3.txt"]

CODIGO = [
    # motor y entorno
    "run_headless_v2.py", "motor_evolutivo.py", "organismo.py",
    "entorno_banco.py", "entorno_evaluacion_fijo.py",
    # entorno_dificil.py es el entorno legacy, pero banco_tareas.py y
    # entorno_evaluacion_fijo.py importan de él Tarea y los validadores, así que es
    # dependencia real del motor vigente y no material histórico.
    "entorno_dificil.py", "llm_backends.py",
    "banco_tareas.py", "dificultad.py", "calibrar_banco.py",
    "parallel_ollama_backend.py", "validacion_semantica.py", "juez_llm.py",
    # medición y análisis
    "estilo.py", "guardar_figura.py",
    "auditoria_contraste.py", "analiz_multiseed.py", "analiz_escalera.py",
    "analiz_gemma.py", "retro_medir_banco.py",
    # figuras del paper
    "graficar_multiseed.py", "graficar_comparativa.py", "graficar_cross_substrato.py",
    "graficar_escalera.py", "graficar_fijacion_multiseed.py", "graficar_termometro.py",
]

MODELFILES = ["modelfiles_gemma", "modelfiles_escalera"]

CALIBRACIONES = [
    "calibracion_qwen3_0.6b.json", "calibracion_qwen3_1.7b.json",
    "calibracion_qwen3_8b.json", "calibracion_gemma4_12b.json",
]

# Las 28 ejecuciones del paper. Quedan fuera las de prueba (test_*) y las legacy
# anteriores al banco calibrado (M1_AxisQwen, M3_NexusQwen, M4_BioLlamaQwen,
# C1_Qwen3_Base), que no entran en ninguna tabla ni figura.
RUNS = [
    # bloque Qwen3-1.7B, tres seeds (15)
    "results/MIA/AxisQwen/M1_AxisQwen_Banco",
    "results/MIA/AxisQwen/M1_AxisQwen_Banco_s101",
    "results/MIA/AxisQwen/M1_AxisQwen_Banco_s102",
    "results/MIA/AxisQwen/A3_AxisQwen_noIM_Banco",
    "results/MIA/AxisQwen/A3_AxisQwen_noIM_Banco_s101",
    "results/MIA/AxisQwen/A3_AxisQwen_noIM_Banco_s102",
    "results/MIA/NexusQwen/M3_NexusQwen_Banco",
    "results/MIA/NexusQwen/M3_NexusQwen_Banco_s101",
    "results/MIA/NexusQwen/M3_NexusQwen_Banco_s102",
    "results/MIA/BioLlamaQwen/M4_BioLlamaQwen_Banco",
    "results/MIA/BioLlamaQwen/M4_BioLlamaQwen_Banco_s101",
    "results/MIA/BioLlamaQwen/M4_BioLlamaQwen_Banco_s102",
    "results/Modelo Base/Qwen3 1.7B/C1_Qwen3_Banco",
    "results/Modelo Base/Qwen3 1.7B/C1_Qwen3_Banco_s101",
    "results/Modelo Base/Qwen3 1.7B/C1_Qwen3_Banco_s102",
    # escalera de escala (8)
    "results/MIA/AxisQwen06b/E06_AxisQwen06b_s100",
    "results/MIA/AxisQwen06b/E06_AxisQwen06b_s101",
    "results/MIA/AxisQwen06b/E06_AxisQwen06b_s102",
    "results/MIA/AxisQwen8b/E8_AxisQwen8b_s100",
    "results/MIA/AxisQwen8b/E8_AxisQwen8b_s101",
    "results/MIA/AxisQwen8b/E8_AxisQwen8b_s102",
    "results/Modelo Base/Qwen3 0.6B/Cb06_Qwen3_06B_s100",
    "results/Modelo Base/Qwen3 8B/Cb8_Qwen3_8B_s100",
    # bloque Gemma4-12B, seed única por diseño (5)
    "results/MIA/AxisGemma/Mg1_AxisGemma_Banco",
    "results/MIA/AxisGemma/Ag3_AxisGemma_noIM_Banco",
    "results/MIA/NexusGemma/Mg3_NexusGemma_Banco",
    "results/MIA/BioLlamaGemma/Mg4_BioLlamaGemma_Banco",
    "results/Modelo Base/Gemma4 12B/Cg_Gemma4_Banco",
]

# Artefactos por ejecución que se publican.
#   campeones_por_generacion.json  el campeón de cada generación: sustrato de toda
#                                  medida de prevalencia (lo leen los scripts)
#   termometro.jsonl               cada medición en el held-out (lo leen los scripts)
#   config_experiment.json         los parámetros con que corrió, tal como se escribieron
#   metrics.csv                    la serie de entrenamiento
#   timing.json                    respalda el presupuesto de cómputo de §3.6
#   best_organism.txt              el genoma final en texto plano
# Quedan fuera: checkpoint_motor.pkl (33 MB de pickles, y un repo público no debería
# pedirle a nadie que haga unpickle), linaje.json y pool_genetico_final.json (ningún
# script publicado los lee), campeones.json y pool_genetico.json (duplicados byte a
# byte de los anteriores), metrics_gen*.csv (cortes incrementales) y los PNG por
# ejecución (regenerables).
ARTEFACTOS = ["campeones_por_generacion.json", "termometro.jsonl",
              "config_experiment.json", "metrics.csv", "timing.json",
              "best_organism.txt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", required=True, type=Path)
    ap.add_argument("--destino", default=Path("."), type=Path)
    a = ap.parse_args()
    org, dst = a.origen.resolve(), a.destino.resolve()

    for f in SPECS + CODIGO:
        shutil.copy2(org / f, dst / f)
    print(f"especificaciones de persona: {len(SPECS)}   módulos: {len(CODIGO)}")

    for d in MODELFILES:
        shutil.rmtree(dst / d, ignore_errors=True)
        shutil.copytree(org / d, dst / d)
    (dst / "artifacts").mkdir(exist_ok=True)
    for c in CALIBRACIONES:
        shutil.copy2(org / "artifacts" / c, dst / "artifacts" / c)
    print(f"modelfiles: {len(MODELFILES)} directorios   calibraciones: {len(CALIBRACIONES)}")

    n_run = n_art = faltan = 0
    for run in RUNS:
        (dst / run).mkdir(parents=True, exist_ok=True)
        n_run += 1
        for art in ARTEFACTOS:
            src = org / run / art
            if src.exists():
                shutil.copy2(src, dst / run / art)
                n_art += 1
            else:
                faltan += 1
    print(f"ejecuciones: {n_run}   artefactos copiados: {n_art}"
          + (f"   (ausentes en origen: {faltan})" if faltan else ""))
    print(f"destino: {dst}")


if __name__ == "__main__":
    main()
