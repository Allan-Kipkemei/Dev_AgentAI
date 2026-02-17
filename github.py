import time
from typing import Any, Dict, List, Optional

import requests

from utils import require_env


def gh_headers() -> Dict[str, str]:
    token = require_env("GITHUB_TOKEN")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "devagent-pr-review-cli",
    }


def gh_request(method: str, url: str, params: Optional[dict] = None, json_body: Optional[dict] = None) -> Any:
    # light retry for rate limits / transient errors
    for attempt in range(1, 6):
        r = requests.request(
            method=method,
            url=url,
            headers=gh_headers(),
            params=params,
            json=json_body,
            timeout=40,
        )

        if r.status_code == 403 and "rate limit" in r.text.lower():
            time.sleep(2 ** attempt)
            continue

        if r.status_code >= 400:
            raise RuntimeError(f"GitHub API error {r.status_code}: {r.text}")

        return r.json()

    raise RuntimeError("GitHub API retry exhausted (rate limits/transient failures).")


def fetch_pr(owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
    return gh_request("GET", f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}")


def fetch_pr_files(owner: str, repo: str, pr_number: int, max_files: int = 60) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    page = 1
    per_page = 100

    while len(files) < max_files:
        batch = gh_request(
            "GET",
            f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files",
            params={"page": page, "per_page": per_page},
        )
        if not batch:
            break

        files.extend(batch)

        if len(batch) < per_page:
            break

        page += 1

    return files[:max_files]


def post_pr_comment(owner: str, repo: str, pr_number: int, body: str) -> None:
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    for attempt in range(1, 6):
        r = requests.post(url, headers=gh_headers(), json={"body": body}, timeout=40)

        if r.status_code == 403 and "rate limit" in r.text.lower():
            time.sleep(2 ** attempt)
            continue

        if r.status_code >= 400:
            raise RuntimeError(f"Failed to post comment {r.status_code}: {r.text}")

        return

    raise RuntimeError("Posting comment retry exhausted.")
