# Persona Imprinting

[![DOI](https://zenodo.org/badge/1409651587.svg)](https://doi.org/10.5281/zenodo.24357580)

Code and data for *Persona Imprinting: Evolved Prompts Inherit the Identity of the Model that
Writes Them* (Castillo-Sepúlveda, Torres-Yévenes & Lanas-Ocampo, Axis Dynamics SpA).

A genetic algorithm evolves a **structured prompt genome** over locally served language models.
Two of the four genes, the *role* and the reasoning *strategies*, are authored by the model under
study itself, which makes every act of variation a transmission event. When that model is
conditioned by a **persona specification** (the `SYSTEM` block of an Ollama model file), the
persona writes itself into the population's inheritable material: persona-specific symbolic
markers appear in 6 of 6 persona-conditioned runs and 0 of 6 control and ablation runs (Fisher
exact *p* = 0.002, run as the experimental unit). Every marker lands in *strategies*, none in
*role*. Switching off model-authored gene synthesis removes the effect in every replicate, which
identifies the route of entry.

This repository holds the engine, the task bank, the persona specifications, the analysis scripts
and the complete per-run artifacts for all **28 runs** in the paper. Every table and figure is
regenerated here by a named script, from logged data.

Nothing is withheld or transformed. The persona specifications are published in full, including
the base64-encoded strings whose verbatim reappearance in evolved genomes is the central finding,
so the detectors of §3.6 can be re-derived from the specifications rather than taken on trust.
See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

## Reproduce the results

No GPU and no model server are needed: the analysis reads the run artifacts in `results/`.

```bash
pip install scipy matplotlib

python3 auditoria_contraste.py     # §4.1 central contrast: 6/6 vs 0/6, threshold curve,
                                   #      detector specificity, background rate
python3 analiz_multiseed.py        # §4.1 Qwen3-1.7B block, three seeds
python3 analiz_escalera.py         # §4.3 scale ladder, 0.6B / 1.7B / 8B
python3 analiz_gemma.py            # §4.2 Gemma4-12B block
```

Figures, written to `artifacts/` as both PNG and vector PDF:

```bash
python3 graficar_multiseed.py            # Figure 1  multiseed_imprinting
python3 graficar_fijacion_multiseed.py   # Figure 2  glyph_trajectory_multiseed
python3 graficar_comparativa.py          # Figure 3  qwen_block_overview
python3 graficar_cross_substrato.py      # Figure 4  cross_substrate_imprinting
python3 graficar_escalera.py             # Figure 5  scale_ladder
```

`auditoria_contraste.py` is the one to read first. It is the single traceable source for the
central contrast: it defines the event before counting it, treats the run as the experimental
unit, and prints the full threshold-sensitivity curve rather than a chosen threshold.

## Re-run the experiments

Needs two GPUs' worth of Ollama and about 25 h per run at 1.7B (104 h at 12B). See
`requirements.txt` for the fuller dependency set.

```bash
# One Ollama server pinned per GPU. Vulkan off: otherwise a server grabs both GPUs and hangs.
CUDA_VISIBLE_DEVICES=0 OLLAMA_VULKAN=false OLLAMA_CONTEXT_LENGTH=4096 \
  OLLAMA_HOST=127.0.0.1:11434 ollama serve
CUDA_VISIBLE_DEVICES=1 OLLAMA_VULKAN=false OLLAMA_CONTEXT_LENGTH=4096 \
  OLLAMA_HOST=127.0.0.1:11435 ollama serve

# Mount a persona. The .txt specifications are complete Ollama model files.
ollama create AxisQwen -f axis_qwen_3.1.txt

# Calibrate difficulty for a substrate. Required before --usar_banco; difficulty is
# measured empirically per substrate and levels are NOT comparable across substrates.
python3 calibrar_banco.py --modelo qwen3:1.7b --K 5 --puertos 11434,11435

# A canonical run (persona + model-authored operator)
python3 run_headless_v2.py --tipo MIA --modelo AxisQwen \
  --generaciones 1000 --poblacion 10 --evaluaciones 1 \
  --usar_banco --banco_modelo qwen3:1.7b --seed 100 \
  --experiment_name M1_AxisQwen_Banco --puertos 11434,11435 \
  --termometro_cada 25 --checkpoint_interval 25

# The ablation arm: same thing with model-authored gene synthesis switched off
python3 run_headless_v2.py ... --no_intelligent_mutation
```

`--banco_modelo` names the **base substrate**, not the persona: calibration is a property of the
substrate. Runs resume idempotently with `--resume <run_dir>`.

## What is where

| | |
|---|---|
| `run_headless_v2.py` | CLI and orchestration; the persona catalogue lives in `get_model_config_v2()` |
| `motor_evolutivo.py` | the GA. `_sintetizar_gen()` is the model-authored operator — the causal route |
| `organismo.py` | `Genotipo.compilar_prompt()`, where the genome becomes a prompt |
| `entorno_banco.py` | the environment in use; samples the training partition, unlocks difficulty bins |
| `entorno_evaluacion_fijo.py` | the fixed held-out battery — measurement, never selection |
| `banco_tareas.py` | 139 tasks, 10 categories × 5 levels, each with its validator |
| `estilo.py` | the style instrument: glyphs, persona lexicons, systematicity. **Measured, never in fitness** |
| `auditoria_contraste.py` | the central contrast, end to end |
| `*.txt` (3) | the persona specifications, which are complete Ollama model files |
| `results/<type>/<model>/<run>/` | per run: `campeones_por_generacion.json`, `termometro.jsonl`, `config_experiment.json`, `metrics.csv`, `timing.json`, `best_organism.txt` |

Two design choices worth knowing before changing anything. Prompts are compiled with
`<start_of_turn>` tags for **all** substrates, deliberately, to keep runs comparable; changing it
breaks comparability with every run here. And the published runs use `--evaluaciones 1`, which is
the dominant noise source in the capability metric and the reason capability is read as a run mean
rather than per generation.

### A note on names

`BioLlama` and `biollama` appear throughout run identifiers, directory names and stored
measurement fields. The persona is called **BIO** in the paper; `BioLlama` is a legacy label from
an early version of the project that ran on Llama 3. Those strings are left untouched because they
are the keys under which the measurements were written — renaming them would mean either editing
stored results or breaking the path from the paper back to the logs. The specification file is
`biovex_deep_q3.txt` throughout.

Code comments and the specifications are in Spanish; the study was conducted in Spanish, which
§6 of the paper discusses as a limitation.

## Citing

Please cite the paper:

> Castillo-Sepúlveda, J., Torres-Yévenes, M., & Lanas-Ocampo, J. C. (2026). *Persona Imprinting:
> Evolved Prompts Inherit the Identity of the Model that Writes Them.* Under review, *Artificial
> Life* (MIT Press).

and, for this code and data release:

> Castillo-Sepúlveda, J., Torres-Yévenes, M., & Lanas-Ocampo, J. C. (2026). *Persona Imprinting:
> code and data* (Version 1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.24357581

`10.5281/zenodo.24357581` is this release; `10.5281/zenodo.24357580` always
resolves to the most recent one.

## License

[MIT](LICENSE). Run artifacts are released under the same terms. Correspondence:
[jorge.castillo@axisdynamics.cl](mailto:jorge.castillo@axisdynamics.cl).
