# On the Role of Citations in Preference Data

Code and data for **[On the Role of Citations in Preference Data](https://arxiv.org/abs/2608.21376)** (EMNLP 2026).

## Setup

Use Python 3.10+ and R. Run all commands from the repository root.

```bash
python -m pip install -r requirements.txt
Rscript install_packages.R
```

R is needed for the regression analyses. Collecting LLM judgments requires [vLLM](https://docs.vllm.ai) and GPUs; vLLM is installed separately.

## Reproduce the main results

**Regression analyses.** Refit the models and recompute the estimates used in Figures 2–4 and Tables 2–3:

```bash
Rscript sciarena/mem.R
Rscript researchqa/mem.R
```

Each command runs the human model and all four LLM-judge models. To run a subset, append judge names, e.g. `Rscript researchqa/mem.R human qwen`. Outputs go to each dataset's `results/mem/`: `ame.csv` contains average marginal effects in percentage points, and `<judge>.txt` contains model summaries and diagnostics. The SciArena human model also writes `model_ranking.csv`.

**Citation perturbations.** This command uses the included judgments; no model fitting or GPU is needed.

```bash
# Citation-perturbation analysis (§4.3)
python -m researchqa.perturbation
```

Perturbation statistics are written to [`researchqa/results/perturbation.csv`](researchqa/results/perturbation.csv).

## Repository layout

```text
common/        Shared judge prompts, inference client, and regression helpers
sciarena/      SciArena feature extraction, judging, data, and results (§3)
researchqa/    ResearchQA pipeline and citation perturbations (§4)
appendix/      Supporting analyses, judgments, and results
```

The [appendix guide](appendix/README.md) covers prompt sensitivity (Table 1), random-effects model selection (Appendix B), and the response-length × citation-diversity interaction check (Limitations). These analyses are optional for reproducing the main results.

## Rebuild features or collect new judgments

The included predictors and judgments are sufficient for the analyses above. To regenerate
them, obtain the original data and follow the steps below.

### 1. Obtain the datasets

[SciArena](https://huggingface.co/datasets/yale-nlp/SciArena) is downloaded automatically.
For ResearchQA, first download the human annotations following the [data instructions](researchqa/data/upstream/README.md). ResearchQA question metadata are downloaded automatically during feature extraction. 
Original question and response texts are not included in this repository.

### 2. Extract predictors

```bash
python -m sciarena.features
python -m researchqa.features
```

These commands download `roberta-large` for BERTScore and replace the feature CSVs in each `data/` directory.

### 3. Run an LLM judge

Judge names are `llama`, `gemma`, `qwen`, and `skywork`; exact model IDs and prompts are in [`common/judge.py`](common/judge.py). 
The paper used four A100 GPUs for Llama and Skywork, and one for Gemma and Qwen.

Start the vLLM server in one terminal, replacing the placeholders below with your judge name, GPU count, and port (e.g. `llama`, `4`, and `8000`):

```bash
python -m common.judge --judge <judge> --gpus <gpus> --port <port>
```

Then collect judgments in another terminal:

```bash
python -m sciarena.run_judge --judge <judge> --port <port>
python -m researchqa.run_judge --judge <judge> --port <port>
python -m researchqa.run_judge --judge <judge> --port <port> --perturbed
```

Judgments use temperature 0 and are appended to each dataset's `results/judgments/` directory, or `researchqa/results/perturbation_judgments/` for `--perturbed`. 
Existing records are skipped when resuming. 
To collect a fresh set, first move the corresponding included CSV to a backup location. 
Fresh perturbation runs include an unperturbed `none` control for every judge; analysis uses it when present.

## Data and result conventions
- **Analysis pairs:** 12,249 for SciArena and 169 for ResearchQA. The SciArena judgment files also include pairs excluded from the regression analysis; the analysis scripts filter them.
- **Predictors:** citation diversity (CD), citation count (CC), citation age (CA; SciArena), citation mismatch (CM; ResearchQA), response length (RL), and response content (RC). Regressions use standardized differences between responses. The exact model formulas are in each dataset's `mem.R`.
- **Judgments:** `choice` is `A` or `B` for the first or second displayed response; `order` identifies the presentation order. Each pair is judged in both orders.
- **ResearchQA response assignment:** `features.csv` uses GPT as Response A. The human model uses `features_shuffled.csv`, with the randomized assignment saved in `response_a_assignment.csv`.
- **Perturbations:** `delta_pp` is the intact-citation preference rate minus the perturbed rate, averaged over both orders within each pair. Positive values indicate a drop in preference. `flipped_pct` is conditional on choosing that response in **both** baseline orders and counts losing that preference in **at least one** perturbed order.

## Citation

```bibtex
@article{hou2026citations,
  title   = {On the Role of Citations in Preference Data},
  author  = {Hou, Yu and Daum{\'e} III, Hal and Rudinger, Rachel and Walden, William},
  journal = {arXiv preprint arXiv:2608.21376},
  year    = {2026}
}
```

Please also cite SciArena ([Zhao et al., 2025](https://arxiv.org/abs/2507.01001)) and ResearchQA ([Yifei et al., 2025](https://arxiv.org/abs/2509.00496)) when using their data.

## License

Code is released under the [MIT License](LICENSE). Datasets and models used in this work remain subject to their respective licenses and terms.
