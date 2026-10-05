"""SciArena data and the fixed-effects predictors (§2, §3.1).

Loads SciArena (Zhao et al., 2025a) from the Hugging Face Hub and, for every
pair with an A/B vote and at least one citation in each response, computes
per response:

    citation count     (CC)  number of in-text citations
    citation diversity (CD)  Shannon entropy (bits) of the cited-source distribution
    citation age       (CA)  mean of ln(2026 - year) over author-year citations
    response length    (RL)  characters

and their A - B differences, plus the response-content difference (RC), from
BERTScore (roberta-large) precision on the citation-stripped responses.

Output: sciarena/data/features.csv (12,249 pairs). A GPU is recommended.

    python -m sciarena.features
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent

# Responses cite sources as "Li et al. 2025a"; each item's citation list maps
# such strings to document ids (a hex hash or "123@456").
DOC_PATTERN = r"[0-9]+@[0-9]+|[a-f0-9]{40}"
AUTHOR_YEAR_PATTERN = r"[a-zA-Z]+ et al\.\s\d{4}[a-z]?"
EMPTY_PARENS_PATTERN = r" \((,\s)*\)."


def load_sciarena(path: Path | None = None) -> list[dict]:
    """SciArena pairs with a clear human preference (vote "A" or "B").

    Reads the `train` split from the Hugging Face Hub, or a local JSON dump of
    it. Of its 13,204 pairs, 12,283 have an A/B vote.
    """
    if path is not None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        from datasets import load_dataset

        data = list(load_dataset("yale-nlp/SciArena", split="train"))
    return [item for item in data if item["vote"] in ("A", "B")]


def replace_citations_with_ids(response: str, citations: list[dict]) -> str:
    citation_to_id = {doc["concise_authors"]: doc["id"] for doc in citations}
    for citation, doc_id in citation_to_id.items():
        response = response.replace(citation, doc_id)
    return response


def cited_sources(response: str, citations: list[dict]) -> list[str]:
    """All in-text citations of a response, as source identifiers (with repeats)."""
    from nltk.tokenize import sent_tokenize

    # Keep "et al." from ending a sentence.
    text = replace_citations_with_ids(response, citations).replace("et al.", "et al#")
    sources = []
    for sent in sent_tokenize(text):
        sent = sent.strip().replace("et al#", "et al.")
        sources += re.findall(DOC_PATTERN, sent)
        # Author-year citations that do not resolve to a listed document.
        sources += re.findall(AUTHOR_YEAR_PATTERN, re.sub(DOC_PATTERN, "", sent))
    return sources


def citation_entropy(sources: list[str]) -> float:
    if not sources:
        return 0
    probs = [n / len(sources) for n in Counter(sources).values()]
    return sum(-p * np.log2(p) for p in probs)


def citation_age(response: str) -> float:
    """Mean ln(2026 - year) over author-year citations (log: years are skewed)."""
    years = [
        int(re.search(r"\d{4}", c).group())
        for c in re.findall(AUTHOR_YEAR_PATTERN, response)
    ]
    if not years:
        return 4.5  # placeholder for responses without author-year citations
    return sum(np.log(2026 - y) for y in years) / len(years)


def strip_citations(response: str, citations: list[dict]) -> str:
    """Remove in-text citations, leaving only the response content."""
    response = replace_citations_with_ids(response, citations)
    response = re.sub(DOC_PATTERN, "", response)
    response = re.sub(AUTHOR_YEAR_PATTERN, "", response)
    response = re.sub(EMPTY_PARENS_PATTERN, ".", response)
    return response.strip()


def unique_content_difference(scorer, a: str, b: str) -> float:
    """RC: share of A's content not covered by B minus the same for B.

    BERTScore precision P(x, y) measures how much of x is covered by y, so the
    share of unique content in x relative to y is 1 - P(x, y).
    """
    p_ab, _, _ = scorer.score([a], [b])
    p_ba, _, _ = scorer.score([b], [a])
    return (1 - p_ab.item()) - (1 - p_ba.item())


def response_features(response: str, citations: list[dict]) -> dict:
    sources = cited_sources(response, citations)
    return {
        "char_length": len(response),
        "num_citations": len(sources),
        "citation_entropy": citation_entropy(sources),
        "citation_years": citation_age(response),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="local JSON dump of SciArena (default: Hugging Face)")
    parser.add_argument("--output", type=Path, default=HERE / "data" / "features.csv")
    args = parser.parse_args()

    import nltk
    import transformers
    from bert_score import BERTScorer

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    transformers.logging.set_verbosity_error()
    scorer = BERTScorer(model_type="roberta-large", lang="en")
    with open(HERE / "data" / "category_mapping.json", encoding="utf-8") as f:
        subject_to_category = json.load(f)

    rows = []
    for item in tqdm(load_sciarena(args.input)):
        a = response_features(item["responseA"], item["citations_a"])
        b = response_features(item["responseB"], item["citations_b"])
        if a["num_citations"] == 0 or b["num_citations"] == 0:
            continue  # both responses must cite at least one source (§3.1)

        rows.append(
            {
                "qid": item["id"],
                "human_vote": item["vote"],
                "response_a_char_length": a["char_length"],
                "response_b_char_length": b["char_length"],
                "citations_a_num_citations": a["num_citations"],
                "citations_b_num_citations": b["num_citations"],
                "citations_a_citation_entropy": a["citation_entropy"],
                "citations_b_citation_entropy": b["citation_entropy"],
                "citation_years_a": a["citation_years"],
                "citation_years_b": b["citation_years"],
                "response_char_length_difference": a["char_length"] - b["char_length"],
                "citation_num_citations_difference": a["num_citations"] - b["num_citations"],
                "citation_entropy_difference": a["citation_entropy"] - b["citation_entropy"],
                "citation_years_difference": a["citation_years"] - b["citation_years"],
                "bert_score_difference": unique_content_difference(
                    scorer,
                    strip_citations(item["responseA"], item["citations_a"]),
                    strip_citations(item["responseB"], item["citations_b"]),
                ),
                "model_a": item["modelA"],
                "model_b": item["modelB"],
                "model_pair": f"{item['modelA']}_{item['modelB']}",
                "question_type": item["question_type"],
                "subject": item["subject"],
                "main_category": subject_to_category[item["subject"]],
                "human_vote_int": int(item["vote"] == "A"),
            }
        )

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} pairs to {args.output}")


if __name__ == "__main__":
    main()
