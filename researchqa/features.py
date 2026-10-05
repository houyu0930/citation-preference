"""ResearchQA data and the fixed-effects predictors (§2, §4.1).

Loads the 169 ResearchQA pairs with a clear human preference: the human
annotations released by Yifei et al. (2025), which you download into
researchqa/data/upstream/ (see the README there), plus question metadata from
the Hugging Face dataset. Every pair compares gpt-4.1-mini ("gpt") with
gemini-2.5-flash ("gemini").

Responses cite sources with numbered markers ("[1]", "[2, 3]") and end with a
reference list of lines "[N] Title (Year) [link](URL)". Per response:

    citation count     (CC)  number of in-text citations
    citation diversity (CD)  Shannon entropy (bits) of the cited-source distribution
    citation mismatch  (CM)  in-text citations with no reference-list entry,
                             plus reference-list entries never cited in the text
    response length    (RL)  characters of the answer (without the reference list)

and their gpt - gemini differences, plus the response-content difference
(RC; BERTScore with roberta-large on the citation-stripped answers).

Outputs (researchqa/data/):
    features.csv           Response A = gpt; used for the LLM-judge models
    features_shuffled.csv  Response A = gpt or gemini at random per pair (§4.1),
                           from response_a_assignment.csv; used for the human model

    python -m researchqa.features
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ANNOTATION_FILES = [
    "nlp_plurality_preferences.json",
    "cv_plurality_preferences.json",
    "multi_domain_plurality_preferences.json",
]
REFERENCE_SEPARATOR = "\n\n\n"
DIFFERENCES = [
    "response_char_length_difference",
    "citation_cnt_difference",
    "citation_diversity_difference",
    "mismatched_citation_cnt_difference",
    "bert_score_difference",
]


def load_items(metadata: bool = False) -> dict[str, dict]:
    """The 169 pairs whose plurality preference is "gpt" or "gemini", by id.

    With `metadata`, adds each question's domain, subdomain, and field.
    """
    items = {}
    for name in ANNOTATION_FILES:
        with open(DATA / "upstream" / name, encoding="utf-8") as f:
            for item in json.load(f):
                if item["plurality_preference"] in ("gpt", "gemini"):
                    items[item["id"]] = item
    if metadata:
        from datasets import load_dataset

        for row in load_dataset("realliyifei/ResearchQA", split="full"):
            if row["id"] in items:
                items[row["id"]]["meta"] = {
                    key: row[key] for key in ("general_domain", "subdomain", "field")
                }
    return items


def answer_part(response: str) -> str:
    return response.split(REFERENCE_SEPARATOR)[0]


def inline_citation_counts(answer: str) -> Counter:
    """How often each reference number is cited in the answer text."""
    counts = Counter()
    for marker in re.finditer(r"\[([\d,\s]+)\]", answer):
        for n in map(int, re.findall(r"\d+", marker.group(1))):
            if n == 2019:
                continue  # a bracketed year, not a citation
            counts[n] += 1
    return counts


def reference_list_ids(response: str) -> list[int]:
    lines = response.split(REFERENCE_SEPARATOR)[1].split("\n")
    return [int(line.split(" ")[0].strip("[]")) for line in lines if line.startswith("[")]


def citation_diversity(counts: Counter) -> float:
    total = sum(counts.values())
    return float(sum(-(n / total) * np.log2(n / total) for n in counts.values()))


def citation_mismatch(counts: Counter, listed: list[int]) -> int:
    not_listed = sum(n for ref, n in counts.items() if ref not in listed)
    not_cited = sum(1 for ref in listed if ref not in counts)
    return not_listed + not_cited


def strip_citations(response: str) -> str:
    return re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", answer_part(response))


def unique_content_difference(scorer, a: str, b: str) -> float:
    """RC: share of A's content not covered by B minus the same for B.

    BERTScore precision P(x, y) measures how much of x is covered by y, so the
    share of unique content in x relative to y is 1 - P(x, y).
    """
    p_ab, _, _ = scorer.score([a], [b])
    p_ba, _, _ = scorer.score([b], [a])
    return (1 - p_ab.item()) - (1 - p_ba.item())


def response_features(response: str) -> dict:
    answer = answer_part(response)
    counts = inline_citation_counts(answer)
    return {
        "response_char_len": len(answer),
        "citation_cnt": sum(counts.values()),
        "citation_diversity": citation_diversity(counts),
        "mismatched_citation_cnt": citation_mismatch(counts, reference_list_ids(response)),
    }


def randomize_response_order(features: pd.DataFrame, assignment: dict) -> pd.DataFrame:
    """Make `assignment[id]` ("gpt" or "gemini") Response A of each pair."""
    shuffled = features.copy()
    shuffled.insert(1, "response_a", shuffled["id"].map(assignment))
    flip = shuffled["response_a"] == "gemini"
    shuffled.loc[flip, DIFFERENCES] = -shuffled.loc[flip, DIFFERENCES]
    shuffled.loc[flip, "human_vote_int"] = 1 - shuffled.loc[flip, "human_vote_int"]
    return shuffled


def main():
    import transformers
    from bert_score import BERTScorer

    transformers.logging.set_verbosity_error()
    scorer = BERTScorer(model_type="roberta-large", lang="en")

    rows = []
    for item in tqdm(load_items(metadata=True).values()):
        gpt = response_features(item["gpt_response"])
        gemini = response_features(item["gemini_response"])
        row = {
            "id": item["id"],
            **item["meta"],
            "human_vote": item["plurality_preference"],
            "human_vote_int": int(item["plurality_preference"] == "gpt"),
        }
        for name in gpt:
            row[f"gpt_{name}"] = gpt[name]
            row[f"gemini_{name}"] = gemini[name]
        for name, column in [
            ("response_char_len", "response_char_length_difference"),
            ("citation_cnt", "citation_cnt_difference"),
            ("citation_diversity", "citation_diversity_difference"),
            ("mismatched_citation_cnt", "mismatched_citation_cnt_difference"),
        ]:
            row[column] = gpt[name] - gemini[name]
        row["bert_score_difference"] = unique_content_difference(
            scorer, strip_citations(item["gpt_response"]), strip_citations(item["gemini_response"])
        )
        rows.append(row)

    features = pd.DataFrame(rows)
    features.to_csv(DATA / "features.csv", index=False)

    assignment_path = DATA / "response_a_assignment.csv"
    if not assignment_path.exists():
        rng = random.Random(0)
        sides = [rng.choice(["gpt", "gemini"]) for _ in rows]
        pd.DataFrame({"id": features["id"], "response_a": sides}).to_csv(assignment_path, index=False)
    assignment = pd.read_csv(assignment_path).set_index("id")["response_a"].to_dict()
    randomize_response_order(features, assignment).to_csv(DATA / "features_shuffled.csv", index=False)
    print(f"Saved features of {len(features)} pairs to {DATA}")


if __name__ == "__main__":
    main()
