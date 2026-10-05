"""Collect LLM-judge preferences on ResearchQA (§4).

Every pair is judged in both presentation orders (`gpt_first`,
`gemini_first`) on the full responses, i.e. including their reference lists.
With --perturbed, the judge sees the citation-perturbed pairs of §4.3 instead,
plus an unperturbed control condition (`none`) to serve as the baseline.

Output: researchqa/results/judgments/{judge}.csv, or with --perturbed
researchqa/results/perturbation_judgments/{judge}.csv.

    python -m common.judge --judge qwen                         # shell 1: vLLM server
    python -m researchqa.run_judge --judge qwen                 # shell 2
    python -m researchqa.run_judge --judge qwen --perturbed     # §4.3
"""

import argparse
from pathlib import Path

from common.judge import JUDGES, PROMPTS, run_judgments
from researchqa.features import load_items
from researchqa.perturbation import PERTURBATIONS, perturbed_records

HERE = Path(__file__).resolve().parent


def build_pairs(records: list[dict], extra_keys: tuple[str, ...] = ()) -> list[dict]:
    """One judge prompt per record and presentation order. Each record needs
    `item_id`, `query`, `gpt_response`, and `gemini_response`; `extra_keys`
    are copied into the output."""
    pairs = []
    for rec in records:
        for order in ("gpt_first", "gemini_first"):
            first, second = rec["gpt_response"], rec["gemini_response"]
            if order == "gemini_first":
                first, second = second, first
            pairs.append(
                {
                    "item_id": rec["item_id"],
                    **{key: rec[key] for key in extra_keys},
                    "order": order,
                    "prompt": PROMPTS["origin"].format(
                        question=rec["query"], response_a=first, response_b=second
                    ),
                }
            )
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", required=True, choices=list(JUDGES))
    parser.add_argument("--perturbed", action="store_true", help="judge the citation-perturbed pairs (§4.3)")
    parser.add_argument("--server", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    items = load_items()
    if args.perturbed:
        records = [
            {"item_id": item_id, "condition": "none", "perturbed": "", **item}
            for item_id, item in sorted(items.items())
        ]
        for condition in PERTURBATIONS:
            records += perturbed_records(items, condition)
        pairs = build_pairs(records, extra_keys=("condition", "perturbed"))
        output = HERE / "results" / "perturbation_judgments" / f"{args.judge}.csv"
        key_fields = ["condition", "item_id", "perturbed", "order"]
    else:
        pairs = build_pairs([{"item_id": item_id, **item} for item_id, item in items.items()])
        output = HERE / "results" / "judgments" / f"{args.judge}.csv"
        key_fields = ["item_id", "order"]

    run_judgments(
        pairs,
        judge=args.judge,
        prompt="origin",
        output_path=output,
        key_fields=key_fields,
        server=args.server,
        port=args.port,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
