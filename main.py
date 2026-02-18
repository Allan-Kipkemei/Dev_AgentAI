import os
import argparse

from dotenv import load_dotenv
load_dotenv()

from utils import parse_pr_url
from github import fetch_pr, fetch_pr_files, post_pr_comment
from review import build_file_context, validate_review_against_allowed_files, format_review
from llm import call_llm_review


def main():
    ap = argparse.ArgumentParser(description="DevAgent: PR Review CLI (minimal)")
    ap.add_argument("pr_url", help="PR URL: https://github.com/<owner>/<repo>/pull/<num>")
    ap.add_argument("--post", action="store_true", help="Post review as a PR comment")
    ap.add_argument("--max-files", type=int, default=60, help="Max PR files to include")
    ap.add_argument("--patch-chars", type=int, default=3500, help="Max chars per patch chunk")
    ap.add_argument("--max-chunks", type=int, default=80, help="Max total patch chunks across all files")
    ap.add_argument("--model", default=os.getenv("DEVAGENT_MODEL", "gpt-4.1-mini"), help="OpenAI model")
    args = ap.parse_args()

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
        print("\n✅ Posted review to PR.")


if __name__ == "__main__":
    main()
