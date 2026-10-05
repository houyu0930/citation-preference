"""Do LLM judges detect invalid citations? Citation perturbations on ResearchQA (§4.3).

For each pair and condition, one of the two responses is picked at random and
its reference list is perturbed, leaving the answer text and its in-text [N]
markers untouched:

    year_future        each year replaced by an impossible future year
    year_random        each year replaced by a random incorrect past year
    title_shuffle      the words within each title shuffled
    title_url_rematch  URLs permuted across the reference list, so each
                       title-URL pair looks plausible but no longer identifies
                       the same paper (needs >= 2 references; 126 of 169 pairs)

`python -m researchqa.run_judge --judge <judge> --perturbed` collects the
judgments. Running this module analyzes them: each pair is scored by how
often, over its two presentation orders, the judge prefers the response whose
references get perturbed (0, 0.5, or 1), and

    delta = preference with intact citations - preference with perturbed citations

averaged over pairs, in percentage points (positive = the judge penalizes the
perturbed citations), with a paired bootstrap 95% CI over pairs and a
Wilcoxon signed-rank test. `flipped` is, of the pairs where the judge
preferred the response in both orders while its citations were intact, the
share for which it chooses the alternative in at least one order after
perturbation (not necessarily a reversal in both orders).

The intact-citation baseline is the `none` condition when the perturbation run
includes it (an unperturbed control judged in the same run; Llama and Skywork),
and otherwise the main-experiment judgments (Gemma and Qwen).

Output: researchqa/results/perturbation.csv

    python -m researchqa.perturbation
"""

import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common.judge import JUDGES

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEED = 42
REFERENCE_SEPARATOR = "\n\n\n"
REFERENCE_RE = re.compile(r"^\[(\d+)\]\s+(.*)\s\((\d{4})\)\s\[link\]\((\S+)\)\s*$")
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 12345

# ---------------------------------------------------------------------------
# Perturbations. Each takes the parsed references of one response, modifies
# them in place, and returns the list of changes.
# ---------------------------------------------------------------------------


def perturb_year_future(refs: list[dict], rng: random.Random) -> list:
    changes = []
    for ref in refs:
        new_year = rng.randint(2027, 2040)
        changes.append((ref["idx"], ref["year"], new_year))
        ref["year"] = new_year
    return changes


def perturb_year_random(refs: list[dict], rng: random.Random) -> list:
    changes = []
    for ref in refs:
        new_year = rng.choice([y for y in range(1950, 2025) if y != ref["year"]])
        changes.append((ref["idx"], ref["year"], new_year))
        ref["year"] = new_year
    return changes


def perturb_title_shuffle(refs: list[dict], rng: random.Random) -> list:
    changes = []
    for ref in refs:
        words = ref["title"].split(" ")
        if len(words) < 2:
            continue  # nothing to shuffle
        shuffled = words[:]
        for _ in range(5):  # avoid returning the original order
            rng.shuffle(shuffled)
            if shuffled != words:
                break
        new_title = " ".join(shuffled)
        changes.append((ref["idx"], ref["title"], new_title))
        ref["title"] = new_title
    return changes


def perturb_title_url_rematch(refs: list[dict], rng: random.Random) -> list:
    if len(refs) < 2:
        return []  # nothing to permute against
    urls = [ref["url"] for ref in refs]
    for _ in range(20):  # draw a derangement: no reference keeps its own URL
        permuted = urls[:]
        rng.shuffle(permuted)
        if all(new != old for new, old in zip(permuted, urls)):
            break
    else:
        return []
    changes = []
    for ref, new_url in zip(refs, permuted):
        changes.append((ref["idx"], ref["url"], new_url))
        ref["url"] = new_url
    return changes


PERTURBATIONS = {
    "year_future": perturb_year_future,
    "year_random": perturb_year_random,
    "title_shuffle": perturb_title_shuffle,
    "title_url_rematch": perturb_title_url_rematch,
}


def perturb_response(response: str, perturb, rng: random.Random) -> tuple[str, list]:
    """Return the response with a perturbed reference list, and the changes."""
    parts = response.split(REFERENCE_SEPARATOR)
    lines = [line for line in parts[1].split("\n") if line.strip()] if len(parts) > 1 else []
    refs = [REFERENCE_RE.match(line) for line in lines]
    if not refs or any(ref is None for ref in refs):
        return response, []
    refs = [dict(zip(("idx", "title", "year", "url"), m.groups())) for m in refs]
    for ref in refs:
        ref["year"] = int(ref["year"])
    changes = perturb(refs, rng)
    references = "\n".join(f"[{r['idx']}] {r['title']} ({r['year']}) [link]({r['url']})" for r in refs)
    return parts[0] + REFERENCE_SEPARATOR + references, changes


def perturbed_records(items: dict[str, dict], condition: str) -> list[dict]:
    """Pairs with one response's references perturbed (pairs where nothing
    could be perturbed are skipped)."""
    rng = random.Random(SEED)
    records = []
    for item_id in sorted(items):
        item = items[item_id]
        side = rng.choice(["gpt", "gemini"])
        responses = {"gpt": item["gpt_response"], "gemini": item["gemini_response"]}
        responses[side], changes = perturb_response(responses[side], PERTURBATIONS[condition], rng)
        if changes:
            records.append(
                {
                    "item_id": item_id,
                    "condition": condition,
                    "perturbed": side,
                    "query": item["query"],
                    "gpt_response": responses["gpt"],
                    "gemini_response": responses["gemini"],
                }
            )
    return records


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def load_judgments(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).dropna(subset=["choice"])
    first = df["order"].str.removesuffix("_first")
    second = np.where(first == "gpt", "gemini", "gpt")
    df["chosen"] = np.where(df["choice"] == "A", first, second)
    return df


def preference_for(df: pd.DataFrame, side: pd.Series) -> pd.Series:
    """Per pair, the share of presentation orders in which `side[pair]` was
    chosen. Pairs without a judgment for both orders are dropped."""
    df = df[df["item_id"].isin(side.index)]
    hits = (df["chosen"] == df["item_id"].map(side)).groupby(df["item_id"])
    return hits.mean()[hits.size() == 2]


def compare(baseline: pd.Series, perturbed: pd.Series) -> dict:
    items = baseline.index.intersection(perturbed.index)
    base, pert = baseline[items].to_numpy(), perturbed[items].to_numpy()
    delta = base - pert

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = delta[rng.integers(0, len(delta), size=(N_BOOTSTRAP, len(delta)))].mean(axis=1)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    preferred = base == 1
    return {
        "n_pairs": len(items),
        "baseline_pct": 100 * base.mean(),
        "perturbed_pct": 100 * pert.mean(),
        "delta_pp": 100 * delta.mean(),
        "ci_low_pp": 100 * ci_low,
        "ci_high_pp": 100 * ci_high,
        "wilcoxon_p": stats.wilcoxon(delta, zero_method="wilcox").pvalue if delta.any() else np.nan,
        "n_preferred_baseline": int(preferred.sum()),
        "flipped_pct": 100 * np.mean(base[preferred] > pert[preferred]),
    }


def main():
    rows = []
    for judge in JUDGES:
        path = RESULTS / "perturbation_judgments" / f"{judge}.csv"
        if not path.exists():
            print(f"Skipping {judge}: no {path.relative_to(HERE.parent)}")
            continue
        judgments = load_judgments(path)
        baseline = judgments[judgments["condition"] == "none"]
        baseline_name = "none"
        if baseline.empty:
            baseline = load_judgments(RESULTS / "judgments" / f"{judge}.csv")
            baseline_name = "main"
        for condition in PERTURBATIONS:
            perturbed = judgments[judgments["condition"] == condition]
            side = perturbed.groupby("item_id")["perturbed"].first()
            result = compare(preference_for(baseline, side), preference_for(perturbed, side))
            rows.append({"judge": judge, "condition": condition, "baseline": baseline_name, **result})

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / "perturbation.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
        print(results.assign(wilcoxon_p=results.wilcoxon_p.map("{:.2g}".format)).to_string(index=False))


if __name__ == "__main__":
    main()
