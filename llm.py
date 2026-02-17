import json
import os
from typing import Any, Dict, List

from pydantic import ValidationError

from utils import require_env
from review import ReviewOutput


def call_llm_review(
    pr: Dict[str, Any],
    file_summaries: List[Dict[str, Any]],
    allowed_files: List[str],
    model: str,
) -> ReviewOutput:
    """
    Calls OpenAI chat completions and returns validated ReviewOutput.
    """
    require_env("OPENAI_API_KEY")

    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    pr_title = pr.get("title", "")
    pr_body = pr.get("body") or ""
    base = pr.get("base", {}).get("ref", "")
    head = pr.get("head", {}).get("ref", "")

    system = (
        "You are a senior software engineer doing a careful GitHub pull request review.\n"
        "Hard rules:\n"
        "- Only mention files listed in allowed_files.\n"
        "- If patch chunks are missing for a file, say you need context and avoid guessing.\n"
        "- Prefer concrete, actionable feedback tied to the diff.\n"
        "- Output MUST be valid JSON and MUST match the schema exactly.\n"
        "- Do NOT include markdown, code fences, or extra commentary.\n"
    )

    payload = {
        "task": "Review this PR. Identify blocking issues, non-blocking improvements, and tests to add.",
        "pr": {
            "title": pr_title,
            "body": pr_body[:6000],
            "base": base,
            "head": head,
        },
        "allowed_files": allowed_files,
        "files": file_summaries,
        "schema": {
            "summary": "string",
            "risk_level": "low|medium|high",
            "blocking_issues": [{"file": "path", "line": 123, "issue": "string", "fix": "string(optional)"}],
            "non_blocking": [{"file": "path", "line": 123, "issue": "string", "fix": "string(optional)"}],
            "tests_to_add": [{"package": "string(optional)", "test_name": "string", "reason": "string"}],
        },
        "review_style": [
            "Call out real bugs, edge cases, security risks, and correctness issues.",
            "Avoid generic advice unless tied to a specific diff chunk.",
            "Be specific about tests: what behavior to test and why.",
        ],
    }

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload)},
        ],
    )

    content = (resp.choices[0].message.content or "").strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Model did not return valid JSON: {e}\n\nRaw:\n{content}")

    try:
        review = ReviewOutput.model_validate(data)
    except ValidationError as e:
        raise RuntimeError(f"Model JSON did not match schema:\n{e}\n\nRaw:\n{content}")

    return review
