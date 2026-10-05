"""LLM judges: models, prompts, answer parsing, and an async client for a vLLM
(OpenAI-compatible) server.

Run as a script to start the vLLM server for a judge (it stays in the
foreground; run the judge scripts from another shell):

    python -m common.judge --judge llama --gpus 4 --port 8000

Judgments are appended to a CSV as they complete, so an interrupted run
resumes when the same command is re-run (judged pairs are skipped).
"""

import argparse
import asyncio
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import requests
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# Hugging Face model ids of the four LLM judges (paper, Appendix A.1).
JUDGES = {
    "llama": "meta-llama/Llama-3.3-70B-Instruct",
    "gemma": "google/gemma-3-27b-it",
    "qwen": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "skywork": "Skywork/Skywork-Critic-Llama-3.1-70B",
}

# Judge prompts, verbatim as run. "origin" is the SciArena judge prompt used in
# all main experiments (§2); "cot" and "length_debiased" are the two variants
# of the prompt-sensitivity analysis (Appendix A.3).
PROMPTS = {
    "origin": """
You are an expert in scientific literature synthesis. Your task is to evaluate the quality of two AI-generated citation-attributed responses to a user's question. Assess both responses for relevance, accuracy, clarity, and appropriate use of citations. Then, select the response, Output (a) or Output (b), that best address the user's question.

User Question:
{question}

Output (a):
{response_a}

Output (b):
{response_b}

Which is best, Output(a) or Output (b)?
Answer with only: "A" or "B"
""".strip(),
    "cot": """
You are an expert in scientific literature synthesis. Your task is to evaluate the quality of two AI-generated citation-attributed responses to a user's question. Assess both responses for relevance, accuracy, clarity, and appropriate use of citations. Then, select the response, Output (a) or Output (b), that best address the user's question.

User Question:
{question}

Output (a):
{response_a}

Output (b):
{response_b}

Which is best, Output (a) or Output (b)?
Let's think step by step, then give your final answer.
End your response with a line formatted exactly as:
Answer: <A or B>
""".strip(),
    "length_debiased": """
You are an expert in scientific literature synthesis. Your task is to evaluate the quality of two AI-generated citation-attributed responses to a user's question. Assess both responses for relevance, accuracy, clarity, and appropriate use of citations. Then, select the response, Output (a) or Output (b), that best address the user's question.

User Question:
{question}

Output (a):
{response_a}

Output (b):
{response_b}

Which is best, Output(a) or Output (b)?
Do NOT allow the length of the responses to influence your evaluation.
Answer with only: "A" or "B"
""".strip(),
}

# Generation budget per prompt: a single letter, or a reasoning chain.
MAX_TOKENS = {"origin": 10, "length_debiased": 10, "cot": 1536}

# "Answer: X" line of a chain-of-thought output. Tolerates "Answer: **B**",
# "Answer: (a)" and "Answer: Output (a)"; the trailing \b keeps
# "**Final Answer:**\nAnswer: B" from matching the "A" of the second "Answer".
ANSWER_LINE_RE = re.compile(
    r"answer\s*:\s*\**\s*(?:output\s*)?\(?\s*([ab])\b", re.IGNORECASE
)


def parse_choice(raw_output: str | None, prompt: str = "origin") -> str | None:
    """Parse the judge's verdict ("A" or "B") from its raw output, else None."""
    if not raw_output:
        return None
    if prompt == "cot":
        matches = ANSWER_LINE_RE.findall(raw_output)
        if matches:
            return matches[-1].upper()
        # Fallback: the last non-empty line is a bare letter.
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        if lines and lines[-1].upper() in ("A", "B"):
            return lines[-1].upper()
        return None
    # Some judges put a verdict on the first line, followed by a rationale.
    # Require a complete verdict there: "Answer: B" is B, while prose such
    # as "Both responses ..." must not silently become a vote for B.
    lines = raw_output.strip().splitlines()
    if not lines:
        return None
    text = lines[0].strip().strip("\"'`*[]()., ")
    text = re.sub(r"^answer\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^output\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("\"'`*[]()., ").upper()
    return text if text in ("A", "B") else None


def check_server(server: str, port: int) -> None:
    """Exit with a helpful message if no vLLM server is reachable."""
    url = f"http://{server}:{port}/v1/models"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        raise SystemExit(
            f"vLLM server not reachable at {url}.\n"
            "Start one first, e.g.: python -m common.judge --judge qwen"
        )
    print(f"Serving: {[m['id'] for m in resp.json().get('data', [])]}")


def _load_done_keys(output_path: Path, key_fields: list[str]) -> set[tuple]:
    if not output_path.exists():
        return set()
    with open(output_path, newline="", encoding="utf-8") as f:
        return {tuple(row[k] for k in key_fields) for row in csv.DictReader(f)}


async def _judge_one(client, model, pair, prompt, semaphore, max_retries=3):
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": pair["prompt"]}],
                    temperature=0.0,
                    max_tokens=MAX_TOKENS[prompt],
                    seed=42,
                )
                raw_output = response.choices[0].message.content
                return pair, raw_output, parse_choice(raw_output, prompt)
            except Exception as e:  # noqa: BLE001 - retry on any API error
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts: {e}")
                    return pair, None, None
                await asyncio.sleep(2**attempt)


async def _run(pairs, model, prompt, output_path, key_fields, api_url, concurrency):
    client = AsyncOpenAI(
        base_url=api_url, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"), timeout=120.0
    )
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [_judge_one(client, model, p, prompt, semaphore) for p in pairs]

    fieldnames = key_fields + ["choice", "raw_output"]
    write_header = not output_path.exists()
    n_failed = 0
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            pair, raw_output, choice = await coro
            if raw_output is None:
                n_failed += 1
                continue  # not written, so a re-run retries it
            row = {k: pair[k] for k in key_fields}
            writer.writerow({**row, "choice": choice or "", "raw_output": raw_output})
            f.flush()
    if n_failed:
        print(f"{n_failed} requests failed; re-run the same command to retry them.")


def run_judgments(
    pairs: list[dict],
    judge: str,
    prompt: str,
    output_path: Path,
    key_fields: list[str],
    server: str = "localhost",
    port: int = 8000,
    concurrency: int = 16,
) -> None:
    """Judge every pair and append (key fields, choice, raw_output) rows to a CSV.

    Each pair is a dict holding the `key_fields` that identify it plus the
    fully formatted `prompt` string. Pairs already in `output_path` are skipped.
    """
    check_server(server, port)
    done = _load_done_keys(output_path, key_fields)
    todo = [p for p in pairs if tuple(str(p[k]) for k in key_fields) not in done]
    print(f"{len(todo)} of {len(pairs)} pairs left to judge -> {output_path}")
    if not todo:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(
        _run(
            todo,
            JUDGES[judge],
            prompt,
            output_path,
            key_fields,
            f"http://{server}:{port}/v1",
            concurrency,
        )
    )


def launch_server():
    """Start a vLLM server for one judge. In the paper we used 4 A100s for
    Llama and Skywork and 1 A100 for Gemma and Qwen (Appendix A.1)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", required=True, choices=list(JUDGES))
    parser.add_argument("--gpus", type=int, default=1, help="tensor-parallel size")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", JUDGES[args.judge],
        "--host", "0.0.0.0",
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--tensor-parallel-size", str(args.gpus),
        "--dtype", "bfloat16",
        "--enforce-eager",
        "--seed", "42",
    ]  # fmt: skip
    print("Launching:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down vLLM server.")


if __name__ == "__main__":
    launch_server()
