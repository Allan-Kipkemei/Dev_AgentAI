import json
import os
import time
from typing import Any

from pydantic import ValidationError

from review import ReviewOutput
from utils import require_env

_MAX_BODY_CHARS = 6_000
_MAX_FILES_CHARS = 40_000
_MAX_RETRIES = 3
_RETRY_DELAY = 2.0


def _build_payload(
    pr: dict[str, Any],
    file_summaries: list[dict[str, Any]],
    allowed_files: list[str],
) -> dict[str, Any]:
    pr_title = pr.get("title", "")
    pr_body = (pr.get("body") or "")[:_MAX_BODY_CHARS]
    base = pr.get("base", {}).get("ref", "")
    head = pr.get("head", {}).get("ref", "")

    # Serialize files, then truncate to a char budget to avoid blowing the context window.
    files_json = json.dumps(file_summaries)
    if len(files_json) > _MAX_FILES_CHARS:
        files_json = files_json[:_MAX_FILES_CHARS] + "... [truncated]"

    return {
        "task": (
            "Review this PR. Identify blocking issues, "
            "non-blocking improvements, and tests to add."
        ),
        "pr": {"title": pr_title, "body": pr_body, "base": base, "head": head},
        "allowed_files": allowed_files,
        "files": files_json,
        "output_schema": ReviewOutput.model_json_schema(),
        "review_style": [
            "Call out real bugs, edge cases, security risks, and correctness issues.",
            "Avoid generic advice unless tied to a specific diff chunk.",
            "Be specific about tests: what behavior to test and why.",
            "Only mention files in allowed_files.",
            "If patch chunks are missing for a file, say you need context — do not guess.",
        ],
    }


_SYSTEM = (
    "You are a senior software engineer performing a careful GitHub pull request review.\n"
    "Your response MUST be a single valid JSON object that conforms exactly to the "
    "provided output_schema. No markdown, no code fences, no extra keys, no commentary."
)


def call_llm_review(
    pr: dict[str, Any],
    file_summaries: list[dict[str, Any]],
    allowed_files: list[str],
    model: str,
) -> ReviewOutput:
    """Call OpenAI chat completions and return a validated ReviewOutput."""
    require_env("OPENAI_API_KEY")

    from openai import OpenAI  # local import keeps startup fast if module unused

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    payload = _build_payload(pr, file_summaries, allowed_files)

    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.2,
                response_format={"type": "json_object"},  # guarantees valid JSON
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": json.dumps(payload)},
                ],
            )
            content = (resp.choices[0].message.content or "").strip()
            data = json.loads(content)
            return ReviewOutput.model_validate(data)

        except json.JSONDecodeError as e:
            last_error = RuntimeError(
                f"[attempt {attempt}] Model did not return valid JSON: {e}\n\nRaw:\n{content}"
            )
        except ValidationError as e:
            formatted = json.dumps(e.errors(), indent=2)
            last_error = RuntimeError(
                f"[attempt {attempt}] Model JSON did not match schema:\n{formatted}"
            )
        except Exception as e:
            last_error = e

        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_DELAY * attempt)

    raise RuntimeError(
        f"LLM review failed after {_MAX_RETRIES} attempts."
    ) from last_error