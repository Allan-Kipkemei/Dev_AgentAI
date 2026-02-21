import argparse
import os

from dotenv import load_dotenv

from github import fetch_pr, fetch_pr_files, post_pr_comment
from llm import call_llm_review
from review import (
    build_file_context,
    format_review,
    validate_review_against_allowed_files,
)
from utils import parse_pr_url

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="DevAgent: PR Review CLI (minimal)")
    parser.add_argument(
        "pr_url", help="PR URL: https://github.com/<owner>/<repo>/pull/<num>"
    )
    parser.add_argument("--post", action="store_true", help="Post review as a PR comment")
    parser.add_argument("--max-files", type=int, default=60, help="Max PR files to include")
    parser.add_argument(
        "--patch-chars", type=int, default=3500, help="Max chars per patch chunk"
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=80,
        help="Max total patch chunks across all files",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("DEVAGENT_MODEL", "gpt-4.1-mini"),
        help="OpenAI model",
    )
    args = parser.parse_args()

    owner, repo, num = parse_pr_url(args.pr_url)

    pr = fetch_pr(owner, repo, num)
    pr_files = fetch_pr_files(owner, repo, num, max_files=args.max_files)

    file_summaries, allowed_files = build_file_context(
        pr_files,
        patch_chunk_chars=args.patch_chars,
        max_total_chunks=args.max_chunks,
    )

    review = call_llm_review(pr, file_summaries, allowed_files, model=args.model)
    validate_review_against_allowed_files(review, allowed_files)

    output = format_review(review)
    print(output)

    if args.post:
        post_pr_comment(owner, repo, num, output)
        print("\n[ok] Posted review to PR.")


if __name__ == "__main__":
    main()
