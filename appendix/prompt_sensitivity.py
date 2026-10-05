"""Prompt sensitivity of the LLM judges on SciArena (Appendix A.3, Table 1).

Compares the judgments under each prompt variant (chain-of-thought, length
debiasing) with those under the original prompt, pair by pair and per
presentation order, using Cohen's kappa and raw agreement. Judgments that
could not be parsed in either run are left out.

Output: appendix/results/prompt_sensitivity.csv

    python -m appendix.prompt_sensitivity
"""

from pathlib import Path

import pandas as pd

from common.judge import JUDGES

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"cot": "CoT", "length_debiased": "LD"}


def load_choices(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["item_id", "order", "choice"]).dropna(subset=["choice"])


def cohen_kappa(x: pd.Series, y: pd.Series) -> float:
    observed = (x == y).mean()
    px, py = (x == "A").mean(), (y == "A").mean()
    expected = px * py + (1 - px) * (1 - py)
    return (observed - expected) / (1 - expected)


def main():
    rows = []
    for judge in JUDGES:
        origin = load_choices(ROOT / "sciarena" / "results" / "judgments" / f"{judge}.csv")
        for variant, label in VARIANTS.items():
            paired = origin.merge(
                load_choices(ROOT / "appendix" / "judgments" / f"{judge}_{variant}.csv"),
                on=["item_id", "order"],
                suffixes=("_origin", "_variant"),
            )
            x, y = paired["choice_origin"], paired["choice_variant"]
            rows.append(
                {
                    "judge": judge,
                    "variant": label,
                    "n": len(paired),
                    "kappa": cohen_kappa(x, y),
                    "agreement": (x == y).mean(),
                }
            )

    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "appendix" / "results" / "prompt_sensitivity.csv", index=False)
    cells = (
        results.kappa.map("{:.2f}".format).str.lstrip("0")
        + " ("
        + (100 * results.agreement).round().astype(int).astype(str)
        + "%)"
    )
    table = results.assign(cell=cells).pivot(index="variant", columns="judge", values="cell")
    print("Cohen's kappa (raw agreement) vs. the original prompt:\n")
    print(table[list(JUDGES)].to_string())


if __name__ == "__main__":
    main()
