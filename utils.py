import os
import re

PR_URL_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<num>\d+)(/.*)?$"
)


def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}")
    return v


def parse_pr_url(url: str) -> tuple[str, str, int]:
    m = PR_URL_RE.match(url.strip())
    if not m:
        raise ValueError(
            "Invalid PR URL. Expected: https://github.com/<owner>/<repo>/pull/<number>"
        )
    return m.group("owner"), m.group("repo"), int(m.group("num"))


def chunk_text(text: str, max_chars: int) -> list[str]:
    """
    Chunk by lines to stay under max_chars. Works well for diffs.
    """
    if not text:
        return []
    lines = text.splitlines(True)
    chunks = []
    cur = ""
    for ln in lines:
        if len(cur) + len(ln) > max_chars and cur:
            chunks.append(cur)
            cur = ""
        cur += ln
    if cur:
        chunks.append(cur)
    return chunks
