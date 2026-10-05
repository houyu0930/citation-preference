"""Collect LLM-judge preferences on SciArena (§2, Appendix A).

Every pair is judged in both presentation orders: `a_first` shows Response A
as Output (a); `b_first` swaps the two. The judge sees all 12,283 pairs with
an A/B vote; sciarena/mem.R keeps the 12,249 used in the paper.

Output: sciarena/results/judgments/{judge}.csv, or for the prompt variants of
Appendix A.3, appendix/judgments/{judge}_{prompt}.csv.

    python -m common.judge --judge qwen                    # shell 1: vLLM server
    python -m sciarena.run_judge --judge qwen              # shell 2
    python -m sciarena.run_judge --judge qwen --prompt cot # Appendix A.3
"""

import argparse
from pathlib import Path

from common.judge import JUDGES, PROMPTS, run_judgments
from sciarena.features import load_sciarena

HERE = Path(__file__).resolve().parent


def build_pairs(items: list[dict], prompt: str) -> list[dict]:
    pairs = []
    for item in items:
        for order in ("a_first", "b_first"):
            first, second = item["responseA"], item["responseB"]
            if order == "b_first":
                first, second = second, first
            pairs.append(
                {
                    "item_id": item["id"],
                    "order": order,
                    "prompt": PROMPTS[prompt].format(
                        question=item["question"], response_a=first, response_b=second
                    ),
                }
            )
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", required=True, choices=list(JUDGES))
    parser.add_argument("--prompt", default="origin", choices=list(PROMPTS))
    parser.add_argument("--server", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--input", type=Path, help="local JSON dump of SciArena (default: Hugging Face)")
    args = parser.parse_args()

    if args.prompt == "origin":
        output = HERE / "results" / "judgments" / f"{args.judge}.csv"
    else:
        output = HERE.parent / "appendix" / "judgments" / f"{args.judge}_{args.prompt}.csv"
    run_judgments(
        build_pairs(load_sciarena(args.input), args.prompt),
        judge=args.judge,
        prompt=args.prompt,
        output_path=output,
        key_fields=["item_id", "order"],
        server=args.server,
        port=args.port,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
