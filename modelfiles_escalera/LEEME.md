# Escalera de escala (familia qwen3)

Mismo ADN AXIS (`axis_qwen_3.1.txt`), solo cambia la línea `FROM`. Sirve para separar
**escala** de **familia**: la comparación qwen-1.7B vs gemma-12B confunde las dos cosas.

| modelfile | substrato | estado |
|---|---|---|
| `AxisQwen06b.modelfile` | qwen3:0.6b | ✅ usable |
| (AxisQwen, en `axis_qwen_3.1.txt`) | qwen3:1.7b | ✅ ya corrido, 3 seeds |
| `AxisQwen8b.modelfile` | qwen3:8b | ✅ usable |
| `AxisQwen4b.modelfile` | qwen3:4b | ❌ **DESCARTADO** |

**Por qué se descarta el 4b:** `qwen3:4b` **no obedece la supresión de thinking** — ni
`"think": false` en el payload ni `/no_think` en el prompt. El razonamiento cae dentro del
campo `response` (1570 caracteres para contestar "391"), así que contaminaría el genoma vía
mutación inteligente y las métricas de estilo/verbosidad. Es el mismo motivo por el que se
descartó deepseek-r1 como substrato. Verificado también con un pull fresco: no es el blob.
0.6b, 1.7b y 8b sí responden limpio (4 tokens).

Los modelos viven en un store propio (`/work/jorge/ollama_models`) porque
`/usr/share/ollama/.ollama/models` es del usuario `ollama` y no es escribible.
Servirlos: `OLLAMA_MODELS=/work/jorge/ollama_models OLLAMA_HOST=127.0.0.1:11436 ollama serve`
