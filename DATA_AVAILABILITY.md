# Data availability

All data and code underlying *Persona Imprinting: Evolved Prompts Inherit the Identity of the
Model that Writes Them* are in this repository, unmodified. Nothing is withheld, redacted or
transformed.

That includes the three **persona specifications** in full — `axis_qwen_3.1.txt`,
`nexusvex_deep_q3.txt` and `biovex_deep_q3.txt`. These are the independent variable of the study,
and they are complete Ollama model files: the `FROM` line, the `SYSTEM` block that constitutes the
persona, and the sampling parameters. The base64-encoded strings inside each `CORE[LOCK]` block —
the material whose verbatim reappearance in evolved genomes is the paper's central finding — are
published as they were used, so the detectors described in §3.6 can be re-derived from the
specifications themselves rather than taken on trust.

## Per-run artifacts

Complete artifacts for all **28 runs** reported in the paper:

| file | what it is |
|---|---|
| `campeones_por_generacion.json` | the champion genome of every generation — the substrate of every prevalence measure |
| `termometro.jsonl` | every held-out measurement (40 per run, one every 25 generations) |
| `config_experiment.json` | the parameters the run actually used, as written at launch |
| `metrics.csv` | the training-side fitness series |
| `timing.json` | wall-clock data, behind the compute budget in §3.6 |
| `best_organism.txt` | the final genome as plain text |

## Reproducing the published numbers

No GPU, no model server, no downloads. `pip install scipy matplotlib`, then:

```bash
python3 auditoria_contraste.py     # §4.1 central contrast: 6/6 vs 0/6, p = 0.002,
                                   #      threshold curve, specificity, background rate
python3 analiz_multiseed.py        # §4.1 Qwen3-1.7B block, three seeds
python3 analiz_escalera.py         # §4.3 scale ladder, 0.6B / 1.7B / 8B
python3 analiz_gemma.py            # §4.2 Gemma4-12B block
```

Every table and figure in the paper comes from these scripts and the files above. There is no
intermediate spreadsheet and no number typed in by hand.

`auditoria_contraste.py` is the one to read first: it defines the event before counting it, treats
the run rather than the champion as the experimental unit, and prints the whole
threshold-sensitivity curve instead of a selected threshold.

## Two properties of the detector, stated because they are checkable here

- **Two payloads share a 24-character prefix.** One in AXIS and one in BIO begin with the same
  plaintext phrase, so their base64 encodings are identical for the first 24 characters — which is
  the prefix length the §3.6 detector matches on. It therefore does not distinguish those two. No
  published number is affected, because each run is only ever tested against the specification of
  its own persona, but the limitation is real and now directly verifiable from the files.
- **Replication is not perfectly faithful.** Some champions carry a payload copy containing a
  substitution error, and some carry a copy concatenated with the label of the following field in
  the specification. Heredity here comes with mutation. The variants are in
  `campeones_por_generacion.json` and can be counted directly.

## What is not here, and why

- `checkpoint_motor.pkl` — engine checkpoints, 33 MB of Python pickles. Excluded as a matter of
  policy: a public repository should not ask anyone to unpickle it. Runs resume from these, but no
  published result depends on one.
- `linaje.json`, `pool_genetico_final.json` — lineage and final gene pool. No published analysis
  script reads them. Available on request.
- Per-generation metric snapshots (`metrics_gen*.csv`) and per-run plots, both regenerable.
- **The proposal stream before selection, which was never recorded.** This one is a genuine
  limitation rather than a release decision, and §4.4 of the paper turns on it: without the pool of
  genes the operator *offered*, as against those that survived, the four candidate explanations of
  the scale pattern cannot be told apart. Logging it is the first item in §7.

Correspondence: [jorge.castillo@axisdynamics.cl](mailto:jorge.castillo@axisdynamics.cl).
