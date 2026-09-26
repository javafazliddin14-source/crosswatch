"""Publish the website + demo to a Hugging Face Space.

    pip install huggingface_hub
    hf auth login                       # once, with a token that has write access
    python deploy/huggingface/publish.py --space <user>/crosswatch --repo-url https://github.com/<user>/crosswatch

Fills the website's Links section with the real URLs and writes the README team table from team.json
(commit both changes to GitHub afterwards), then uploads
only what the site needs (code, weights, website); large files go through LFS automatically.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
INCLUDE = ["solution.py", "requirements.txt", "predictions_samples.json", "src", "weights", "website"]
IGNORE = ["__pycache__", "*.pyc", "download.sh"]

SPACE_README = """---
title: CrossWatch
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

Traffic event detection for one intersection camera (WIUT Hackathon 2026, CV track).
Code: {repo_url}
"""


def set_links(repo_url: str, space: str) -> None:
    """Point the website's Links section at the real repository and site."""
    path = ROOT / "website" / "static" / "data" / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    repo_url = repo_url.rstrip("/")
    report["links"] = {
        "Repository": repo_url,
        "Weights": f"{repo_url}/tree/main/weights",
        "predictions_samples.json": f"{repo_url}/blob/main/predictions_samples.json",
        "Website and live demo": f"https://huggingface.co/spaces/{space}",
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


def write_team_table() -> None:
    """README 'Team' section from website/static/data/team.json (single source of truth)."""
    team = json.loads((ROOT / "website" / "static" / "data" / "team.json").read_text(encoding="utf-8"))
    rows = ["| Member | Role | Did | Links |", "|---|---|---|---|"]
    for m in team["members"]:
        links = ", ".join(f"[{k}]({u})" for k, u in m.get("links", {}).items() if u)
        rows.append(f"| {m['name']} | {m['role']} | {m['did']} | {links} |")
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- team:start -->", "<!-- team:end -->"
    i, j = text.index(start) + len(start), text.index(end)
    readme.write_text(text[:i] + "\n" + "\n".join(rows) + "\n" + text[j:], encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="<user>/<space-name>")
    ap.add_argument("--repo-url", required=True, help="https://github.com/<user>/<repo>")
    args = ap.parse_args()

    team = json.loads((ROOT / "website" / "static" / "data" / "team.json").read_text(encoding="utf-8"))
    if any(m["name"].startswith("Member ") for m in team["members"]):
        raise SystemExit("Fill in website/static/data/team.json first (names, roles, what each person did, links).")
    set_links(args.repo_url, args.space)
    write_team_table()
    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in INCLUDE:
            src = ROOT / name
            if src.is_dir():
                shutil.copytree(src, stage / name, ignore=shutil.ignore_patterns(*IGNORE))
            else:
                shutil.copy2(src, stage / name)
        shutil.copy2(HERE / "Dockerfile", stage / "Dockerfile")
        (stage / "README.md").write_text(SPACE_README.format(repo_url=args.repo_url), encoding="utf-8")
        api.upload_folder(folder_path=str(stage), repo_id=args.space, repo_type="space",
                          commit_message="Deploy website and demo")
    print(f"Space: https://huggingface.co/spaces/{args.space}")
    print(f"Direct site URL (for SPACE_URL): https://{args.space.replace('/', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()
