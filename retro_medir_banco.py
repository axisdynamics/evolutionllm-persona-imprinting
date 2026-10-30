#!/usr/bin/env python3
"""
Retro-medición de campeones contra el HELD-OUT del banco ampliado (paso 3 del plan).

De-riesgo barato (~20 min): confirma que el banco nuevo DISCRIMINA entre base e
identidad ANTES de gastar GPU re-corriendo experimentos. Usa el subconjunto held-out
(que la evolución nunca verá) como termómetro, y reporta el perfil por CATEGORÍA y
por NIVEL — no un solo escalar.

Uso:
    evolution/bin/python retro_medir_banco.py "<RUN_DIR>" [--cada N] [--K K]

Ejemplos:
    evolution/bin/python retro_medir_banco.py "results/Modelo Base/Qwen3 1.7B/C1_Qwen3_Base" --cada 25 --K 3
    evolution/bin/python retro_medir_banco.py "results/MIA/AxisQwen/M1_AxisQwen" --cada 25 --K 3

Salida: <RUN_DIR>/retro_banco.json (curva + perfil por categoría/nivel)
"""
import json, sys, argparse, hashlib, re
import requests
from pathlib import Path
from collections import defaultdict
import statistics as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from organismo import Genotipo
from banco_tareas import construir_banco, particion_train_heldout
from entorno_evaluacion_fijo import EntornoEvaluacionFijo
from parallel_ollama_backend import ParallelOllamaBackend, OllamaServer
from dificultad import (cargar_calibracion, particion_por_dificultad,
                        dificultad_continua, bin_dificultad)

MODELO = {
    "Qwen3 1.7B": "qwen3:1.7b", "AxisQwen": "AxisQwen:latest",
    "NexusQwen": "NexusQwen:latest", "BioLlamaQwen": "BioLlamaQwen:latest",
    "Deepseek-R1 8B": "deepseek-r1:8b", "AxisDeep": "AxisDeep:latest",
    "NexusDeep": "NexusDeep:latest", "BioLlamaDeep": "BioLlamaDeep:latest",
}

def geno_hash(g): return hashlib.md5(json.dumps(g, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
def cat_de(tid): return re.match(r'(.+)_L(\d+)_', tid).group(1)
def niv_de(tid): return int(re.match(r'(.+)_L(\d+)_', tid).group(2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--cada", type=int, default=25)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--puertos", default="11434,11435",
                    help="puertos Ollama separados por coma (1 por GPU). Default usa 2 GPUs.")
    ap.add_argument("--frac_heldout", type=float, default=0.4)
    ap.add_argument("--bins", type=int, default=7, help="bins de dificultad para reportar")
    args = ap.parse_args()

    run = Path(args.run_dir)
    cfg = json.load(open(run / "config_experiment.json"))
    modelo = MODELO.get(cfg.get("modelo", ""))
    if not modelo:
        raise SystemExit(f"Modelo '{cfg.get('modelo')}' no mapeado.")
    camps = json.load(open(run / "campeones_por_generacion.json"))

    banco = construir_banco()
    calib = cargar_calibracion(modelo)
    if calib:
        _, heldout = particion_por_dificultad(banco, calib, frac_heldout=args.frac_heldout, n_bins=args.bins)
        print(f"Run {run.name} | modelo {modelo} | held-out: {len(heldout)} | "
              f"DIFICULTAD CALIBRADA ({len(calib)} tareas) | {args.bins} bins")
    else:
        _, heldout = particion_train_heldout(banco, frac_heldout=args.frac_heldout)
        print(f"Run {run.name} | modelo {modelo} | held-out: {len(heldout)} | "
              f"⚠️ sin calibración → niveles NOMINALES (corre calibrar_banco.py primero)")

    sel = camps[::args.cada]
    if sel and sel[-1]["generacion"] != camps[-1]["generacion"]:
        sel.append(camps[-1])
    unicos = {}
    for c in sel:
        unicos.setdefault(geno_hash(c["genotipo"]), c["genotipo"])

    # Detectar GPUs/puertos vivos (paralelismo real entre múltiples Ollama)
    servers = []
    for i, p in enumerate(int(x) for x in args.puertos.split(",")):
        try:
            if requests.get(f"http://localhost:{p}/api/tags", timeout=5).status_code == 200:
                servers.append(OllamaServer("localhost", p, gpu_id=i))
            else:
                print(f"  ⚠️ puerto {p} no responde (status != 200), se omite")
        except Exception:
            print(f"  ⚠️ puerto {p} no responde, se omite")
    if not servers:
        raise SystemExit("Ningún servidor Ollama responde.")
    ngpu = len(servers)
    nprompts = len(unicos) * len(heldout) * args.K
    print(f"Puntos: {len(sel)} | genotipos únicos: {len(unicos)} | ~{nprompts} prompts")
    print(f"Usando {ngpu} GPU(s) (puertos {[s.port for s in servers]}) | ~{nprompts/(2.0*ngpu)/60:.0f} min estimado\n")

    backend = ParallelOllamaBackend(model_name=modelo, servers=servers,
        disable_thinking=True, temperature=0.7)
    backend.validate_all()
    env = EntornoEvaluacionFijo(backend, K=args.K, tareas=heldout)

    res_hash = {}
    for i, (h, g) in enumerate(unicos.items(), 1):
        r = env.evaluar_genotipo(Genotipo.from_dict(g))
        res_hash[h] = r
        print(f"  [{i}/{len(unicos)}] fitness_banco={r['fitness']:.4f}")

    curva = []
    for c in sel:
        r = res_hash[geno_hash(c["genotipo"])]
        # agregados por categoría y por BIN DE DIFICULTAD CALIBRADA (no nominal)
        porcat, porbin = defaultdict(list), defaultdict(list)
        for tid, v in r["desglose"].items():
            porcat[cat_de(tid)].append(v["mean"])
            p = calib.get(tid)
            b = bin_dificultad(dificultad_continua(p), args.bins) if p is not None else niv_de(tid)
            porbin[b].append(v["mean"])
        curva.append({
            "generacion": c["generacion"], "fitness_run": c["fitness"],
            "fitness_banco": r["fitness"],
            "por_categoria": {k: round(st.mean(v), 3) for k, v in porcat.items()},
            "por_dificultad_bin": {b: round(st.mean(v), 3) for b, v in sorted(porbin.items())},
            "estilo": {k: round(v, 3) for k, v in r.get("estilo", {}).items()},
            "estilo_por_categoria": {
                cat: {k: round(v, 2) for k, v in idx.items()}
                for cat, idx in r.get("estilo_por_categoria", {}).items()
            },
        })

    out = run / "retro_banco.json"
    json.dump({"run": run.name, "modelo": modelo, "K": args.K, "cada": args.cada,
               "n_heldout": len(heldout), "curva": curva}, open(out, "w"),
              indent=2, ensure_ascii=False)
    ff = [p["fitness_banco"] for p in curva]
    print(f"\n💾 {out}")
    print(f"fitness_banco: media={st.mean(ff):.3f} min={min(ff):.3f} max={max(ff):.3f} ultimo={ff[-1]:.3f}")

if __name__ == "__main__":
    main()
