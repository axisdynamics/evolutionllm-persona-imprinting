#!/usr/bin/env python3
# run_headless_v2.py
# VERSIÓN ACTUALIZADA PARA EXPERIMENTOS DE LARGA DURACIÓN
# 
# Cambios respecto a run_headless.py:
# - Soporte para ParallelOllamaBackend (2 GPUs)
# - Checkpoints configurables (cada N generaciones)
# - Desactivación de thinking mode para Qwen3/Deepseek-R1
# - Nuevos modelos AXIS (AxisQwen, AxisDeep)
# - Configuración optimizada para 1000 generaciones
# - Mejor logging y monitoreo

import argparse
import os
import time
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import re
from datetime import datetime
import json
import sys

# Configurar Matplotlib para entornos sin pantalla
matplotlib.use('Agg')


def verify_gpu(require_gpu: bool = True) -> dict:
    """Verifica disponibilidad y estado de la GPU."""
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": [],
        "using_gpu": False
    }
    
    print("\n" + "="*60)
    print("🔍 VERIFICACIÓN DE GPU")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA NO disponible")
        if require_gpu:
            print("\n⛔ ABORTANDO: Se requiere GPU")
            sys.exit(1)
        return gpu_info
    
    gpu_info["device_count"] = torch.cuda.device_count()
    
    for i in range(gpu_info["device_count"]):
        props = torch.cuda.get_device_properties(i)
        gpu_info["devices"].append({
            "id": i,
            "name": props.name,
            "memory_gb": round(props.total_memory / (1024**3), 2)
        })
        print(f"  GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f} GB)")
    
    gpu_info["using_gpu"] = True
    print("="*60 + "\n")
    return gpu_info


def set_global_seed(seed: int):
    """Fija seed para reproducibilidad completa."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"🎲 Seed global fijado: {seed}")


# Importaciones del Framework
from motor_evolutivo import MotorEvolutivoProfesional, EvolutionConfig
from entorno_dificil import EntornoDificil
from parallel_ollama_backend import ParallelOllamaBackend, OllamaServer
# Banco calibrado como presión selectiva + termómetro held-out (paso 4)
from banco_tareas import construir_banco
from entorno_banco import EntornoBanco, medir_termometro
from entorno_evaluacion_fijo import EntornoEvaluacionFijo
from dificultad import cargar_calibracion, particion_por_dificultad

# Importar backends originales para compatibilidad
from llm_backends import (
    OllamaAPIBackend, validate_backend, MatchedPromptBackend
)


def get_model_config_v2():
    """
    Configuración de modelos actualizada para experimentos AXIS.
    Incluye modelos base y entidades VEX/MIA.
    """
    return {
        # ============================================
        # MODELOS BASE (Control)
        # ============================================
        "Modelo Base": {
            # Qwen3 1.7b - modelo base para AxisQwen
            "Qwen3 1.7B": {
                "backend": "parallel",
                "model_name": "qwen3:1.7b",
                "disable_thinking": True
            },
            # Deepseek-R1 8b - modelo base para AxisDeep
            "Deepseek-R1 8B": {
                "backend": "parallel",
                "model_name": "deepseek-r1:8b",
                "disable_thinking": True
            },
            # Llama 3 8B (del paper original)
            "Llama3 8B": {
                "backend": "parallel",
                "model_name": "llama3:8b",
                "disable_thinking": False
            },
            # Escalera de ESCALA dentro de la familia qwen3 (0.6B / 1.7B / 8B), para separar
            # escala de familia. Viven en un store propio (/work/jorge/ollama_models) porque
            # /usr/share/ollama/.ollama/models no es escribible: servirlos requiere
            # OLLAMA_MODELS apuntando ahí (ver modelfiles_escalera/LEEME.md).
            # ⚠️ qwen3:4b queda FUERA: no obedece la supresión de thinking (como deepseek-r1).
            "Qwen3 0.6B": {
                "backend": "parallel",
                "model_name": "qwen3:0.6b",
                "disable_thinking": True
            },
            "Qwen3 8B": {
                "backend": "parallel",
                "model_name": "qwen3:8b",
                "disable_thinking": True
            },
            # Gemma4 12B - 2º substrate (thinking se apaga con think:false -> disable_thinking=True)
            "Gemma4 12B": {
                "backend": "parallel",
                "model_name": "gemma4:12b",
                "disable_thinking": True
            },
        },
        
        # ============================================
        # ENTIDADES MIA (Monolithic Identity Architecture)
        # ============================================
        "MIA": {
            # === AXIS (ADN VEX v6.0) ===
            "AxisQwen": {
                "backend": "parallel",
                "model_name": "AxisQwen:latest",
                "disable_thinking": True,
                "substrate": "qwen3:1.7b",
                "architecture": "MIA",
                "identity": "AXIS"
            },
            "AxisDeep": {
                "backend": "parallel",
                "model_name": "AxisDeep:latest",
                "disable_thinking": True,
                "substrate": "deepseek-r1:8b",
                "architecture": "MIA",
                "identity": "AXIS"
            },
            # === AXIS en la escalera de escala — MISMO ADN que AxisQwen, solo cambia el FROM.
            # Aísla ESCALA de FAMILIA: qwen-1.7B vs gemma-12B confunde las dos cosas (§6).
            "AxisQwen06b": {
                "backend": "parallel",
                "model_name": "AxisQwen06b:latest",
                "disable_thinking": True,
                "substrate": "qwen3:0.6b",
                "architecture": "MIA",
                "identity": "AXIS"
            },
            "AxisQwen8b": {
                "backend": "parallel",
                "model_name": "AxisQwen8b:latest",
                "disable_thinking": True,
                "substrate": "qwen3:8b",
                "architecture": "MIA",
                "identity": "AXIS"
            },
            # === NEXUS (sobre Qwen3 y Deepseek) ===
            "NexusQwen": {
                "backend": "parallel",
                "model_name": "NexusQwen:latest",
                "disable_thinking": True,
                "substrate": "qwen3:1.7b",
                "architecture": "MIA",
                "identity": "NEXUS"
            },
            "NexusDeep": {
                "backend": "parallel",
                "model_name": "NexusDeep:latest",
                "disable_thinking": True,
                "substrate": "deepseek-r1:8b",
                "architecture": "MIA",
                "identity": "NEXUS"
            },
            # === BioLlama (sobre Qwen3 y Deepseek) ===
            "BioLlamaQwen": {
                "backend": "parallel",
                "model_name": "BioLlamaQwen:latest",
                "disable_thinking": True,
                "substrate": "qwen3:1.7b",
                "architecture": "MIA",
                "identity": "BioLlama"
            },
            "BioLlamaDeep": {
                "backend": "parallel",
                "model_name": "BioLlamaDeep:latest",
                "disable_thinking": True,
                "substrate": "deepseek-r1:8b",
                "architecture": "MIA",
                "identity": "BioLlama"
            },
            # === Identidades sobre Gemma4 12B (2º substrate; thinking off vía think:false) ===
            "AxisGemma": {
                "backend": "parallel", "model_name": "AxisGemma:latest",
                "disable_thinking": True, "substrate": "gemma4:12b",
                "architecture": "MIA", "identity": "AXIS"
            },
            "NexusGemma": {
                "backend": "parallel", "model_name": "NexusGemma:latest",
                "disable_thinking": True, "substrate": "gemma4:12b",
                "architecture": "MIA", "identity": "NEXUS"
            },
            "BioLlamaGemma": {
                "backend": "parallel", "model_name": "BioLlamaGemma:latest",
                "disable_thinking": True, "substrate": "gemma4:12b",
                "architecture": "MIA", "identity": "BioLlama"
            },
            # === Modelos legacy (Llama3) - si están disponibles ===
            "Nexus Llama3": {
                "backend": "parallel",
                "model_name": "Nexus:latest",
                "disable_thinking": False,
                "substrate": "llama3:8b",
                "architecture": "MIA"
            },
            "BioLlama": {
                "backend": "parallel",
                "model_name": "BioLL:latest",
                "disable_thinking": False,
                "substrate": "llama3:8b",
                "architecture": "MIA"
            },
        },
    }


def create_backend(config: dict, args) -> ParallelOllamaBackend:
    """Crea el backend apropiado según la configuración."""
    
    if config.get("backend") == "parallel":
        # Detectar SOLO puertos Ollama vivos (auto-omite GPUs colgadas)
        import requests as _rq
        puertos = [int(x) for x in getattr(args, "puertos", "11434,11435").split(",")]
        servers = []
        for i, p in enumerate(puertos):
            try:
                if _rq.get(f"http://localhost:{p}/api/tags", timeout=5).status_code == 200:
                    servers.append(OllamaServer("localhost", p, gpu_id=i))
                else:
                    print(f"   ⚠️ puerto {p} no responde, se omite")
            except Exception:
                print(f"   ⚠️ puerto {p} no responde, se omite")
        if not servers:
            raise RuntimeError("Ningún servidor Ollama responde en los puertos indicados.")
        print(f"   Usando {len(servers)} GPU(s): puertos {[s.port for s in servers]}")

        backend = ParallelOllamaBackend(
            model_name=config["model_name"],
            servers=servers,
            timeout=args.timeout,
            disable_thinking=config.get("disable_thinking", True),
            temperature=args.temperature
        )
        return backend
    else:
        # Fallback a OllamaAPIBackend simple
        return OllamaAPIBackend(model_name=config["model_name"])


def sanitize_filename(name: str) -> str:
    """Limpia un nombre para usarlo como nombre de archivo."""
    return re.sub(r'[^\w\-_\. ]', '_', name).strip()


def setup_results_dir(tipo: str, modelo: str, experiment_name: str = None) -> str:
    """Crea y retorna el directorio de resultados."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_model = sanitize_filename(modelo)
    
    if experiment_name:
        base_path = os.path.join("results", tipo, sanitized_model, experiment_name)
    else:
        base_path = os.path.join("results", tipo, sanitized_model, timestamp)
    
    os.makedirs(base_path, exist_ok=True)
    return base_path


def save_checkpoint(motor, save_path: str, gen: int, metrics_df: pd.DataFrame):
    """Guarda checkpoint completo."""
    motor.guardar_estado(save_path)
    
    # Guardar métricas parciales
    metrics_df.to_csv(os.path.join(save_path, f"metrics_gen{gen}.csv"), index=False)
    
    # Guardar estado resumible
    with open(os.path.join(save_path, "resume_state.json"), "w") as f:
        json.dump({
            "last_generation": gen,
            "timestamp": datetime.now().isoformat()
        }, f)


def save_plots(metrics_df: pd.DataFrame, save_path: str):
    """Genera y guarda los gráficos de métricas."""
    # Plot de Fitness
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(metrics_df['Generation'], metrics_df['Max Fitness'], 
            label='Max Fitness', marker='o', linewidth=2, markersize=3)
    ax.plot(metrics_df['Generation'], metrics_df['Avg Fitness'], 
            label='Avg Fitness', linestyle='--', linewidth=2)
    ax.set_title('Evolución del Fitness', fontsize=14)
    ax.set_xlabel('Generación', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'plot_fitness.png'), dpi=150)
    plt.close()

    # Plot de Diversidad
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(metrics_df['Generation'], metrics_df['Diversity'], 
            label='Diversidad Genotípica', color='orange', linewidth=2)
    ax.set_title('Evolución de la Diversidad', fontsize=14)
    ax.set_xlabel('Generación', fontsize=12)
    ax.set_ylabel('Distancia Promedio', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'plot_diversity.png'), dpi=150)
    plt.close()

    # Plot de Especies
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(metrics_df['Generation'], metrics_df['Species Count'], 
            label='Num Especies', color='green', linewidth=2)
    ax.set_title('Dinámica de Especies', fontsize=14)
    ax.set_xlabel('Generación', fontsize=12)
    ax.set_ylabel('Cantidad', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'plot_species.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="EvolutionLLM v2 - Experimentos de larga duración con 2 GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Experimento base Qwen3 1.7B (control)
  python run_headless_v2.py --tipo "Modelo Base" --modelo "Qwen3 1.7B" --generaciones 1000

  # Experimento entidad AxisQwen (MIA)
  python run_headless_v2.py --tipo "MIA" --modelo "AxisQwen" --generaciones 1000

  # Ablación sin intelligent mutation
  python run_headless_v2.py --tipo "MIA" --modelo "AxisQwen" --no_intelligent_mutation

  # Resumir experimento interrumpido
  python run_headless_v2.py --resume results/MIA/AxisQwen/20260322_120000
        """
    )
    
    # === ARGUMENTOS PRINCIPALES ===
    parser.add_argument("--tipo", type=str, required=True,
                        choices=["Modelo Base", "MIA"],
                        help="Tipo de entidad")
    parser.add_argument("--modelo", type=str, required=True,
                        help="Nombre del modelo (ver --list_models)")
    parser.add_argument("--list_models", action="store_true",
                        help="Listar modelos disponibles y salir")
    
    # === PARÁMETROS EVOLUTIVOS ===
    parser.add_argument("--generaciones", "-g", type=int, default=1000,
                        help="Número de generaciones (default: 1000)")
    parser.add_argument("--poblacion", "-p", type=int, default=10,
                        help="Tamaño de la población (default: 10)")
    parser.add_argument("--evaluaciones", "-e", type=int, default=1,
                        help="Evaluaciones por organismo (default: 1)")
    # === BANCO CALIBRADO (paso 4) ===
    parser.add_argument("--usar_banco", action="store_true",
                        help="Usar el banco calibrado como presión selectiva (Reina Roja sobre dificultad) + termómetro held-out")
    parser.add_argument("--banco_modelo", type=str, default=None,
                        help="Modelo para cargar calibración del banco (default: model_name del substrate)")
    parser.add_argument("--n_tareas_gen", type=int, default=8,
                        help="Tareas muestreadas por generación del rango activo (default: 8)")
    parser.add_argument("--bin_inicial", type=int, default=2,
                        help="Bin de dificultad inicial desbloqueado (default: 2)")
    parser.add_argument("--termometro_cada", type=int, default=25,
                        help="Medir el campeón en el held-out cada N generaciones (default: 25)")
    parser.add_argument("--termometro_K", type=int, default=3,
                        help="Evaluaciones por tarea en el termómetro (default: 3)")
    parser.add_argument("--puertos", type=str, default="11434,11435",
                        help="Puertos Ollama (1 por GPU), separados por coma. Default usa 2 GPUs.")
    parser.add_argument("--tasa_mutacion", type=float, default=0.8,
                        help="Probabilidad de mutación (default: 0.8)")
    parser.add_argument("--tasa_cruce", type=float, default=0.7,
                        help="Probabilidad de cruce (default: 0.7)")
    parser.add_argument("--elites", type=int, default=1,
                        help="Número de élites a conservar (default: 1)")
    
    # === REPRODUCIBILIDAD Y ABLACIONES ===
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed para reproducibilidad")
    parser.add_argument("--no_intelligent_mutation", action="store_true",
                        help="Desactivar mutación inteligente (ablación)")
    parser.add_argument("--prob_mutacion_inteligente", type=float, default=0.30,
                        help="Probabilidad de mutación inteligente (default: 0.30)")
    parser.add_argument("--matched_prompt", type=str, default=None,
                        choices=["axis_mia"],
                        help="Inyectar prompt de identidad como system prompt (control)")
    
    # === CHECKPOINTS Y RESUMIR ===
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="Guardar checkpoint cada N generaciones (default: 10)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Ruta para reanudar experimento")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Nombre del experimento (para organizar resultados)")
    
    # === CONFIGURACIÓN DE BACKEND ===
    parser.add_argument("--timeout", type=int, default=180,
                        help="Timeout por request en segundos (default: 180)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperatura de sampling (default: 0.7)")
    parser.add_argument("--skip_validation", action="store_true",
                        help="Saltar validación de backend")
    parser.add_argument("--allow_cpu", action="store_true",
                        help="Permitir ejecución sin GPU")
    
    args = parser.parse_args()
    
    # === LISTAR MODELOS ===
    MODEL_CONFIG = get_model_config_v2()
    
    if args.list_models:
        print("\n📋 MODELOS DISPONIBLES:")
        print("="*60)
        for tipo, modelos in MODEL_CONFIG.items():
            print(f"\n{tipo}:")
            for nombre, config in modelos.items():
                substrate = config.get("substrate", "N/A")
                arch = config.get("architecture", "Base")
                print(f"  - {nombre} (substrate: {substrate}, arch: {arch})")
        return 0
    
    # === VERIFICAR GPU ===
    gpu_info = verify_gpu(require_gpu=not args.allow_cpu)
    
    # === FIJAR SEED ===
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
    set_global_seed(args.seed)
    
    # === VALIDAR MODELO ===
    if args.tipo not in MODEL_CONFIG:
        print(f"❌ Tipo '{args.tipo}' no válido. Usa --list_models")
        return 1
    
    if args.modelo not in MODEL_CONFIG[args.tipo]:
        print(f"❌ Modelo '{args.modelo}' no existe en '{args.tipo}'")
        print("Modelos disponibles:")
        for k in MODEL_CONFIG[args.tipo].keys():
            print(f"  - {k}")
        return 1
    
    model_config = MODEL_CONFIG[args.tipo][args.modelo]
    
    # === CONFIGURAR DIRECTORIO ===
    if args.resume:
        save_path = args.resume
        print(f"♻️ Reanudando desde: {save_path}")
    else:
        save_path = setup_results_dir(args.tipo, args.modelo, args.experiment_name)
        print(f"📁 Directorio de resultados: {save_path}")
    
    # === CREAR BACKEND ===
    print(f"\n🔄 Inicializando backend paralelo...")
    print(f"   Modelo: {model_config['model_name']}")
    print(f"   Thinking mode: {'ON' if not model_config.get('disable_thinking', True) else 'OFF'}")
    
    try:
        backend = create_backend(model_config, args)
        
        if not args.skip_validation:
            print("   Validando servidores...")
            if not backend.validate_all():
                print("\n❌ No todos los servidores Ollama están disponibles.")
                print("Ejecuta en terminales separadas:")
                print("  T1: export OLLAMA_MODELS=/usr/share/ollama/.ollama/models && CUDA_VISIBLE_DEVICES=0 ollama serve")
                print("  T2: export OLLAMA_MODELS=/usr/share/ollama/.ollama/models && CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=0.0.0.0:11435 ollama serve")
                return 1
        
        print("✅ Backend listo")
    except Exception as e:
        print(f"❌ Error inicializando backend: {e}")
        return 1
    
    # === CONFIGURAR MOTOR EVOLUTIVO ===
    print(f"\n🧬 Configurando motor evolutivo...")
    
    # Determinar prob_mutacion_inteligente
    if args.no_intelligent_mutation:
        prob_mut_inteligente = 0.0
        print("   ⚠️ ABLACIÓN: Mutación inteligente DESACTIVADA")
    else:
        prob_mut_inteligente = args.prob_mutacion_inteligente
    
    # Con el banco: SIN ponderación por dificultad (causaba el artefacto del "colapso").
    evolution_config = EvolutionConfig(
        num_evaluaciones_fitness=args.evaluaciones,
        ponderar_fitness_por_dificultad=not args.usar_banco,
        prob_mutacion_inteligente=prob_mut_inteligente
    )

    # === ENTORNO: banco calibrado (paso 4) o EntornoDificil clásico ===
    env_termometro = None
    heldout = []
    if args.usar_banco:
        banco_modelo = args.banco_modelo or model_config["model_name"]
        calib = cargar_calibracion(banco_modelo)
        if not calib:
            print(f"❌ No hay calibración para '{banco_modelo}'. Corre calibrar_banco.py primero.")
            return 1
        banco = construir_banco()
        train, heldout = particion_por_dificultad(banco, calib, frac_heldout=0.4)
        entorno = EntornoBanco(backend, train, calib, n_tareas_gen=args.n_tareas_gen,
                               bin_inicial=args.bin_inicial, seed=args.seed)
        env_termometro = EntornoEvaluacionFijo(backend, K=args.termometro_K, tareas=heldout)
        print(f"🏦 Banco calibrado: {len(train)} train (presión) / {len(heldout)} held-out (termómetro) "
              f"| calib '{banco_modelo}' | n_tareas_gen={args.n_tareas_gen}, bin_inicial={args.bin_inicial}")
    else:
        entorno = EntornoDificil(backend)
    
    motor = None
    gen_inicio = 0
    
    if args.resume:
        checkpoint_path = os.path.join(args.resume, "checkpoint_motor.pkl")
        if os.path.exists(checkpoint_path):
            motor = MotorEvolutivoProfesional.cargar_estado(checkpoint_path, entorno)
            gen_inicio = motor.generacion_actual
            print(f"   ♻️ Reanudando desde generación {gen_inicio}")
        else:
            print("❌ No se encontró checkpoint.")
            return 1
    else:
        motor = MotorEvolutivoProfesional(
            tamaño_poblacion=args.poblacion,
            tasa_mutacion=args.tasa_mutacion,
            tasa_cruce=args.tasa_cruce,
            num_elites=args.elites,
            entorno=entorno,
            config=evolution_config
        )
        motor.iniciar_poblacion()
    
    # === GUARDAR CONFIGURACIÓN ===
    if not args.resume:
        config_data = {
            "timestamp": datetime.now().isoformat(),
            "seed": args.seed,
            "tipo": args.tipo,
            "modelo": args.modelo,
            "model_config": model_config,
            "generaciones": args.generaciones,
            "poblacion": args.poblacion,
            "evaluaciones": args.evaluaciones,
            "tasa_mutacion": args.tasa_mutacion,
            "tasa_cruce": args.tasa_cruce,
            "elites": args.elites,
            "intelligent_mutation_enabled": not args.no_intelligent_mutation,
            "prob_mutacion_inteligente": prob_mut_inteligente,
            "checkpoint_interval": args.checkpoint_interval,
            "usar_banco": args.usar_banco,
            "banco_modelo": (args.banco_modelo or model_config["model_name"]) if args.usar_banco else None,
            "n_tareas_gen": args.n_tareas_gen if args.usar_banco else None,
            "bin_inicial": args.bin_inicial if args.usar_banco else None,
            "ponderar_fitness": not args.usar_banco,
            "gpu_info": gpu_info
        }
        with open(os.path.join(save_path, "config_experiment.json"), "w") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    # === BUCLE EVOLUTIVO ===
    print(f"\n{'='*70}")
    print(f"🚀 INICIANDO EVOLUCIÓN")
    print(f"   Generaciones: {gen_inicio} → {args.generaciones}")
    print(f"   Población: {args.poblacion}")
    print(f"   Checkpoints cada: {args.checkpoint_interval} generaciones")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    timing_por_generacion = []
    
    # Preparar DataFrame para métricas
    metrics_data = []
    if args.resume and motor.historial_metricas:
        for m in motor.historial_metricas:
            metrics_data.append({
                "Generation": m[0],
                "Max Fitness": m[1],
                "Avg Fitness": m[2],
                "Diversity": m[3],
                "Species Count": m[4]
            })
    
    try:
        for gen in range(gen_inicio, args.generaciones):
            gen_start = time.time()
            
            # Evolucionar
            motor.evolucionar()
            
            gen_elapsed = time.time() - gen_start
            timing_por_generacion.append({
                "generacion": gen,
                "tiempo_segundos": gen_elapsed
            })
            
            # Registrar métricas
            metricas = motor.historial_metricas[-1]
            metrics_data.append({
                "Generation": metricas[0],
                "Max Fitness": metricas[1],
                "Avg Fitness": metricas[2],
                "Diversity": metricas[3],
                "Species Count": metricas[4]
            })
            
            # Mostrar progreso
            elapsed_total = time.time() - start_time
            gens_done = gen - gen_inicio + 1
            gens_remaining = args.generaciones - gen - 1
            eta = (elapsed_total / gens_done) * gens_remaining if gens_done > 0 else 0
            
            print(f"Gen {gen+1:4d}/{args.generaciones} | "
                  f"Fit: {metricas[1]:.4f} (avg: {metricas[2]:.4f}) | "
                  f"Div: {metricas[3]:.2f} | "
                  f"Spp: {metricas[4]} | "
                  f"⏱️ {gen_elapsed:.1f}s | "
                  f"ETA: {eta/3600:.1f}h")
            
            # Termómetro held-out: medir el campeón cada N gen (NO afecta la selección)
            if args.usar_banco and env_termometro is not None and (gen + 1) % args.termometro_cada == 0:
                try:
                    campeon = motor.historial_campeones[-1]
                    med = medir_termometro(env_termometro, campeon.genotipo, entorno.calib, generacion=gen + 1)
                    with open(os.path.join(save_path, "termometro.jsonl"), "a", encoding="utf-8") as f:
                        f.write(json.dumps(med, ensure_ascii=False) + "\n")
                    print(f"   🌡️ Termómetro gen {gen+1}: fitness_heldout={med['fitness_heldout']:.3f} "
                          f"| bin_activo={entorno.bin_activo}/{entorno.n_bins}")
                except Exception as e:
                    print(f"   ⚠️ termómetro falló: {e}")

            # Checkpoint
            if (gen + 1) % args.checkpoint_interval == 0:
                df = pd.DataFrame(metrics_data)
                save_checkpoint(motor, save_path, gen + 1, df)
                print(f"   💾 Checkpoint guardado (gen {gen+1})")
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️ Interrupción detectada. Guardando estado...")
        df = pd.DataFrame(metrics_data)
        save_checkpoint(motor, save_path, gen, df)
        print(f"   💾 Estado guardado. Reanudar con: --resume {save_path}")
        return 0
    
    # === POST-PROCESAMIENTO ===
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✨ EVOLUCIÓN COMPLETADA")
    print(f"   Tiempo total: {total_time/3600:.2f} horas")
    print(f"{'='*70}\n")
    
    # Guardar métricas finales
    df = pd.DataFrame(metrics_data)
    df.to_csv(os.path.join(save_path, "metrics.csv"), index=False)
    
    # Mejor organismo
    best = sorted(motor.historial_campeones, key=lambda x: x.fitness, reverse=True)[0]
    with open(os.path.join(save_path, "best_organism.txt"), "w", encoding="utf-8") as f:
        f.write(f"ID: {best.id}\n")
        f.write(f"FITNESS: {best.fitness}\n")
        f.write(f"GENERACION: {best.generacion}\n")
        f.write(f"GENOTIPO:\n{json.dumps(best.genotipo.to_dict(), indent=2, ensure_ascii=False)}\n")
    
    # Plots
    save_plots(df, save_path)
    
    # Timing
    with open(os.path.join(save_path, "timing.json"), "w") as f:
        json.dump({
            "total_segundos": total_time,
            "total_horas": total_time / 3600,
            "promedio_por_generacion_seg": total_time / args.generaciones,
            "por_generacion": timing_por_generacion[-100:]  # Últimas 100 para no ser muy largo
        }, f, indent=2)
    
    # Pool genético final
    with open(os.path.join(save_path, "pool_genetico_final.json"), "w", encoding="utf-8") as f:
        json.dump(motor.pool_genetico.to_dict(), f, indent=2, ensure_ascii=False)
    
    # Campeones por generación
    with open(os.path.join(save_path, "campeones.json"), "w", encoding="utf-8") as f:
        json.dump(motor.campeones_por_generacion, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Resultados guardados en: {save_path}")
    print(f"   - metrics.csv")
    print(f"   - best_organism.txt")
    print(f"   - pool_genetico_final.json")
    print(f"   - campeones.json")
    print(f"   - plot_fitness.png, plot_diversity.png, plot_species.png")
    
    return 0


if __name__ == "__main__":
    exit(main())
