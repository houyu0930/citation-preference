# Supporting analyses

This directory contains the code and saved results for the paper's robustness checks and
model-selection procedure. It is optional for reproducing the main results. Use the setup
in the [main README](../README.md) and run all commands from the repository root.

## Prompt sensitivity (Appendix A.3, Table 1)

Compare the chain-of-thought and length-debiasing prompts with the main SciArena prompt:

```bash
python -m appendix.prompt_sensitivity
```

This uses the included judgments and writes `appendix/results/prompt_sensitivity.csv`, with
Cohen's kappa and raw agreement for each judge and prompt variant. No GPU is required.
`appendix/judgments/` contains the variant judgments; the baseline judgments are in
`sciarena/results/judgments/`. The released chain-of-thought files contain parsed verdicts,
with the reasoning text omitted.

To collect new variant judgments, start the appropriate vLLM server as described in the main
README, move the existing output CSV to a backup location, and run:

```bash
python -m sciarena.run_judge --judge llama --prompt cot
python -m sciarena.run_judge --judge llama --prompt length_debiased
```

## Random-effects model selection (Appendix B)

```bash
Rscript appendix/model_selection.R researchqa human
```

Replace `researchqa` with `sciarena`, and `human` with any judge name. Logs are written to
`appendix/results/model_selection/<dataset>_<judge>_<start>.txt`. The repository includes
ResearchQA selection logs for all five judges.

The procedure retains all fixed effects. While a fit is singular, it first removes
random-effect correlations and then drops the component with the smallest variance.
It subsequently removes the component whose removal most reduces AIC, stopping when no
single removal improves AIC.

- `--start maximal` (default) starts from the maximal structure.
- `--start final` checks whether removing a component from the paper's final structure
  improves AIC; it does not repeat the full selection from the maximal model.

SciArena selection is computationally expensive because each step fits multiple models.
On a multicore Linux or macOS machine, candidate fits can run in parallel:

```bash
Rscript appendix/model_selection.R sciarena human --start final --cores 16
```

## Response-length × citation-diversity interaction (Limitations)

```bash
Rscript appendix/interaction.R researchqa
Rscript appendix/interaction.R sciarena
```

These commands add an RL × CD interaction to each final model. Outputs are written to
`appendix/results/interaction/`: `<dataset>.csv` contains interaction coefficients and
p-values, and `<dataset>_<judge>.txt` contains the model summaries. Saved results for both
datasets are included.

To run a single judge, append its name, e.g. `Rscript appendix/interaction.R sciarena llama`.
SciArena fits can reuse cached final-model estimates as starting values. `--cores` runs
judges in parallel on Linux or macOS.
