#!/usr/bin/env python3
"""
Calibración empírica de dificultad del banco POR SUBSTRATO.

Mide la tasa de acierto del MODELO BASE (con genotipos neutros, fijada ANTES de los
runs evolutivos) sobre las 106 tareas → p_i por tarea. Re-estratifica los niveles por
bandas globales de dificultad medida (en vez de las etiquetas asignadas a mano) y
señala tareas suelo (nadie acierta) y techo (sin headroom) para podar.

IMPORTANTE: la dificultad es relativa al modelo. Hay que correr esto UNA VEZ POR
SUBSTRATO. qwen3 ahora; deepseek-r1:8b antes de correr C2/M2/M5/M6.

Uso (2 GPUs):
    evolution/bin/python calibrar_banco.py --modelo qwen3:1.7b --K 5
    # despues, para deepseek:
    evolution/bin/python calibrar_banco.py --modelo deepseek-r1:8b --K 5

Salida: artifacts/calibracion_<modelo>.json  (p_i, nivel original vs calibrado por tarea)
"""
import json, sys, argparse, re
import requests
from pathlib import Path
from collections import defaultdict
import statistics as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from organismo import Genotipo
from banco_tareas import construir_banco
from entorno_evaluacion_fijo import EntornoEvaluacionFijo
from parallel_ollama_backend import ParallelOllamaBackend, OllamaServer

# Genotipos de REFERENCIA neutros: promediar sobre varios estilos genéricos evita
# que la calibración dependa de un único prompt. NO usan identidad.
REFERENCIA = [
    Genotipo("asistente conciso y directo", "respuesta directa", [], []),
    Genotipo("asistente", "respuesta directa", ["Piensa paso a paso."], []),
    Genotipo("experto que resuelve la tarea", "respuesta directa", ["Verifica tu respuesta."], []),
]

# Bandas de dificultad por tasa de acierto medida (p alto = fácil)
def nivel_calibrado(p: float) -> int:
    if p >= 0.85: return 1
    if p >= 0.65: return 2
    if p >= 0.45: return 3
    if p >= 0.25: return 4
    return 5

def cat_de(tid): return re.match(r'(.+)_L(\d+)_', tid).group(1)
def niv_orig(tid): return int(re.match(r'(.+)_L(\d+)_', tid).group(2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="qwen3:1.7b")
    ap.add_argument("--K", type=int, default=5, help="evaluaciones por tarea por genotipo de referencia")
    ap.add_argument("--puertos", default="11434,11435")
    args = ap.parse_args()

    banco = construir_banco()
    print(f"Calibrando {args.modelo} sobre {len(banco)} tareas | {len(REFERENCIA)} genotipos ref × K={args.K}")

    servers = []
    for i, p in enumerate(int(x) for x in args.puertos.split(",")):
        try:
            if requests.get(f"http://localhost:{p}/api/tags", timeout=5).status_code == 200:
                servers.append(OllamaServer("localhost", p, gpu_id=i))
        except Exception:
            print(f"  ⚠️ puerto {p} no responde, se omite")
    if not servers:
        raise SystemExit("Ningún servidor Ollama responde.")
    nprompts = len(banco) * len(REFERENCIA) * args.K
    print(f"Usando {len(servers)} GPU(s) | ~{nprompts} prompts (~{nprompts/(2.0*len(servers))/60:.0f} min)\n")

    backend = ParallelOllamaBackend(model_name=args.modelo, servers=servers,
        disable_thinking=True, temperature=0.7)
    backend.validate_all()
    env = EntornoEvaluacionFijo(backend, K=args.K, tareas=banco)

    # acumular score medio por tarea a través de los genotipos de referencia
    acc = defaultdict(list)
    for gi, g in enumerate(REFERENCIA, 1):
        print(f"  genotipo referencia {gi}/{len(REFERENCIA)}...")
        r = env.evaluar_genotipo(g)
        for tid, v in r["desglose"].items():
            acc[tid].append(v["mean"])
    p = {tid: st.mean(v) for tid, v in acc.items()}

    # construir reporte por tarea
    tareas = []
    for t in banco:
        pi = p.get(t.id_tarea, 0.0)
        tareas.append({
            "id": t.id_tarea, "categoria": cat_de(t.id_tarea),
            "nivel_original": niv_orig(t.id_tarea),
            "p": round(pi, 3), "nivel_calibrado": nivel_calibrado(pi),
        })

    # --- reporte consola ---
    print("\n=== DISTRIBUCIÓN DE NIVELES: original (asignado a mano) vs calibrado ===")
    cross = defaultdict(int)
    for t in tareas:
        cross[(t["nivel_original"], t["nivel_calibrado"])] += 1
    print("  (filas=original, cols=calibrado)")
    print("        L1c  L2c  L3c  L4c  L5c")
    for lo in range(1, 6):
        fila = "  ".join(f"{cross[(lo,lc)]:3d}" for lc in range(1, 6))
        print(f"  L{lo}o  {fila}")

    suelo = [t for t in tareas if t["p"] < 0.10]
    techo = [t for t in tareas if t["p"] > 0.95]
    print(f"\n⚠️ SUELO (p<0.10, nadie acierta → podar/reformular): {len(suelo)}")
    for t in suelo: print(f"    {t['id']:28s} p={t['p']}")
    print(f"\n⚠️ TECHO (p>0.95, sin headroom → endurecer): {len(techo)}")
    for t in techo: print(f"    {t['id']:28s} p={t['p']}")

    print("\n=== DIFICULTAD MEDIA POR CATEGORÍA (1-p = dificultad) ===")
    porcat = defaultdict(list)
    for t in tareas: porcat[t["categoria"]].append(t["p"])
    for c in sorted(porcat):
        ps = porcat[c]
        print(f"  {c:12s} p_medio={st.mean(ps):.3f}  (rango {min(ps):.2f}-{max(ps):.2f}, n={len(ps)})")

    # guardar SIEMPRE en el artifacts/ del repo (no del CWD), para que dificultad.py lo encuentre
    out = Path(__file__).resolve().parent / "artifacts" / f"calibracion_{args.modelo.replace(':','_').replace('/','_')}.json"
    out.parent.mkdir(exist_ok=True)
    json.dump({"modelo": args.modelo, "K": args.K, "n_referencia": len(REFERENCIA),
               "tareas": tareas}, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\n💾 {out}")

if __name__ == "__main__":
    main()
