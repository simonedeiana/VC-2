"""AlphaEvolve-style evolutionary controller for VC-2 latency research."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import shlex
import shutil
import subprocess
import tempfile
import textwrap
import uuid
from pathlib import Path
from typing import Any

from core import (
    ProgramDatabase,
    apply_edits,
    candidate_cell,
    correctness_matches,
    load_config,
    parse_proposal,
    promising,
)
from evaluate import evaluate, git_output, require_success, run_command


def runtime_paths(root: Path, config: dict[str, Any]) -> tuple[Path, Path]:
    runtime = root / ".autoresearch"
    return runtime, runtime / f"{config['name']}.sqlite3"


def render_result(result: dict[str, Any]) -> str:
    metrics = result.get("metrics", {})
    fps = {key: value for key, value in metrics.items() if key.endswith("_fps")}
    return json.dumps(
        {
            "valid": result.get("valid"),
            "fps": fps,
            "hashes": result.get("hashes", {}),
            "error": result.get("error"),
        },
        indent=2,
        sort_keys=True,
    )


def build_prompt(
    root: Path,
    config: dict[str, Any],
    parent: dict[str, Any],
    inspirations: list[dict[str, Any]],
    island: str,
) -> str:
    background_path = root / "autoresearch" / config["search"].get("background", "BACKGROUND.md")
    background = background_path.read_text(encoding="utf-8") if background_path.is_file() else ""
    prior = []
    for item in inspirations:
        prior.append(
            f"Program {item['id']} ({item['island']}, score={item['score']:.3f})\n"
            f"{render_result(item['result'])}\n"
            f"Proposal summary:\n{item['proposal'][:4000]}"
        )
    return textwrap.dedent(
        f"""
        Act as an expert C++ performance engineer evolving the VC-2 codec.
        The objective is to reduce single-thread encoding and decoding latency while
        preserving exact bitstreams, exact decoded pixels, API compatibility, and all tests.

        Search island: {island}
        Target branch: {config['branch']}

        Current parent result:
        {render_result(parent['result'])}

        Repository-specific background:
        {background}

        Prior high-performing or diverse programs:
        {chr(10).join(prior) if prior else '(none yet)'}

        Propose one coherent experiment. Prefer a focused change that can be evaluated
        independently. Do not edit generated files, benchmark inputs, build directories,
        tests, validators, or the autoresearch evaluator. Correctness gates are absolute.

        Return metadata followed by one or more exact SEARCH/REPLACE blocks:

        TARGET: encoder|decoder|both
        STRATEGY: short-kebab-case-description
        RISK: low|medium|high
        RATIONALE: concise explanation

        FILE: relative/path/to/file.cpp
        <<<<<<< SEARCH
        exact existing text
        =======
        replacement text
        >>>>>>> REPLACE

        Every SEARCH section must match exactly once. Return no Markdown fences.
        """
    ).strip() + "\n"


def proposer_command(config: dict[str, Any], override: str | None) -> list[str]:
    if override:
        return shlex.split(override, posix=os.name != "nt")
    command = config.get("proposer", {}).get("command", [])
    if isinstance(command, str):
        return shlex.split(command, posix=os.name != "nt")
    return [str(part) for part in command]


def invoke_proposer(
    command_template: list[str],
    *,
    worktree: Path,
    prompt: str,
    timeout: int,
) -> str:
    request_dir = worktree / ".autoresearch" / "requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = request_dir / "prompt.txt"
    output_path = request_dir / "proposal.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    replacements = {
        "{worktree}": str(worktree),
        "{prompt_file}": str(prompt_path),
        "{output_file}": str(output_path),
    }
    command = []
    for part in command_template:
        for marker, value in replacements.items():
            part = part.replace(marker, value)
        command.append(part)
    process = run_command(command, cwd=worktree, timeout=timeout)
    require_success(process, "proposal generation")
    if output_path.is_file():
        return output_path.read_text(encoding="utf-8")
    if process.stdout.strip():
        return process.stdout
    raise RuntimeError("proposer produced neither an output file nor stdout")


def add_worktree(repository: Path, path: Path, commit_sha: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    process = run_command(["git", "worktree", "add", "--detach", str(path), commit_sha], cwd=repository, timeout=300)
    require_success(process, "candidate worktree creation")


def remove_worktree(repository: Path, path: Path, temp_root: Path) -> None:
    resolved = path.resolve()
    resolved.relative_to(temp_root.resolve())
    process = run_command(["git", "worktree", "remove", "--force", str(resolved)], cwd=repository, timeout=300)
    if process.returncode and resolved.exists():
        raise RuntimeError(f"failed to remove candidate worktree {resolved}: {process.stdout}")


def record_baseline(root: Path, config: dict[str, Any], tier: str, database: ProgramDatabase) -> dict[str, Any]:
    result = evaluate(root, config, tier)
    if not result["valid"]:
        raise RuntimeError(f"baseline evaluation failed: {result.get('error')}")
    commit_sha = git_output(root, "rev-parse", "HEAD")
    program_id = f"baseline-{commit_sha[:12]}-{tier}"
    database.add(
        program_id=program_id,
        branch=config["branch"],
        parent_id=None,
        generation=0,
        island="baseline",
        commit_sha=commit_sha,
        status="baseline",
        proposal="Human-provided branch baseline",
        metadata={"target": "both", "strategy": "baseline", "risk": "none"},
        result=result,
    )
    for island in config["search"].get("islands", ["default"]):
        database.update_elite(config["branch"], f"{tier}:{island}", program_id)
    return database.get(program_id)


def evaluate_candidate(
    *,
    repository: Path,
    config: dict[str, Any],
    parent: dict[str, Any],
    inspirations: list[dict[str, Any]],
    island: str,
    generation: int,
    command_template: list[str],
    tier: str,
    full_if_promising: bool,
    temp_root: Path,
) -> dict[str, Any]:
    candidate_id = f"g{generation:04d}-{uuid.uuid4().hex[:10]}"
    worktree = temp_root / config["name"] / candidate_id
    proposal = ""
    metadata: dict[str, str] = {"target": "unknown", "strategy": "unknown", "risk": "unknown"}
    result: dict[str, Any] = {"valid": False, "error": "candidate was not evaluated"}
    status = "rejected"
    commit_sha: str | None = None
    try:
        add_worktree(repository, worktree, parent["commit_sha"])
        prompt = build_prompt(worktree, config, parent, inspirations, island)
        proposal = invoke_proposer(
            command_template,
            worktree=worktree,
            prompt=prompt,
            timeout=int(config.get("proposer", {}).get("timeout_seconds", 1800)),
        )
        metadata, edits = parse_proposal(proposal)
        apply_edits(worktree, edits)
        changed = git_output(worktree, "status", "--porcelain")
        if not changed:
            raise RuntimeError("proposal made no repository changes")

        result = evaluate(worktree, config, "smoke")
        if not result["valid"]:
            raise RuntimeError(result.get("error", "smoke evaluation failed"))
        identical, reason = correctness_matches(result, parent["result"])
        if not identical:
            raise RuntimeError(reason)

        result = evaluate(worktree, config, tier)
        if not result["valid"]:
            raise RuntimeError(result.get("error", f"{tier} evaluation failed"))
        identical, reason = correctness_matches(result, parent["result"])
        if not identical:
            raise RuntimeError(reason)
        is_promising, feedback = promising(
            result,
            parent["result"],
            float(config["search"].get("minimum_gain_percent", 0.5)),
            float(config["search"].get("regression_tolerance_percent", 0.5)),
        )
        result["selection_feedback"] = feedback
        if not is_promising:
            raise RuntimeError(feedback)

        if full_if_promising and tier != "full":
            result = evaluate(worktree, config, "full")
            if not result["valid"]:
                raise RuntimeError(result.get("error", "full evaluation failed"))
            identical, reason = correctness_matches(result, parent["result"])
            if not identical:
                raise RuntimeError(reason)
            is_promising, feedback = promising(
                result,
                parent["result"],
                float(config["search"].get("minimum_gain_percent", 0.5)),
                float(config["search"].get("regression_tolerance_percent", 0.5)),
            )
            result["selection_feedback"] = feedback
            if not is_promising:
                raise RuntimeError(feedback)

        git_output(worktree, "add", "--all")
        process = run_command(
            ["git", "commit", "-m", f"autoresearch: {metadata.get('strategy', candidate_id)}"],
            cwd=worktree,
            timeout=120,
        )
        require_success(process, "candidate commit")
        commit_sha = git_output(worktree, "rev-parse", "HEAD")
        git_output(repository, "update-ref", f"refs/autoresearch/{config['name']}/{candidate_id}", commit_sha)
        status = "accepted"
    except Exception as exc:  # Candidate failures are data, not controller failures.
        result.setdefault("valid", False)
        result["valid"] = False
        result["error"] = str(exc)
    finally:
        if worktree.exists():
            remove_worktree(repository, worktree, temp_root)
    return {
        "id": candidate_id,
        "parent_id": parent["id"],
        "generation": generation,
        "island": candidate_cell(metadata, config["search"].get("islands", [island])),
        "commit_sha": commit_sha,
        "status": status,
        "proposal": proposal,
        "metadata": metadata,
        "result": result,
    }


def promote(root: Path, candidate: dict[str, Any]) -> None:
    if candidate["status"] != "accepted" or not candidate["commit_sha"]:
        return
    if git_output(root, "status", "--porcelain"):
        raise RuntimeError("target worktree is dirty; refusing automatic promotion")
    process = run_command(["git", "cherry-pick", candidate["commit_sha"]], cwd=root, timeout=300)
    require_success(process, "candidate promotion")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline_parser = subparsers.add_parser("baseline", help="evaluate and record the current branch")
    baseline_parser.add_argument("--tier", choices=("smoke", "quick", "full"), default="full")

    run_parser = subparsers.add_parser("run", help="generate and evaluate candidate programs")
    run_parser.add_argument("--generations", type=int, default=1)
    run_parser.add_argument("--candidates", type=int, default=2)
    run_parser.add_argument("--parallel", type=int)
    run_parser.add_argument("--tier", choices=("quick", "full"), default="quick")
    run_parser.add_argument("--full-if-promising", action="store_true")
    run_parser.add_argument("--proposer-command")
    run_parser.add_argument("--promote", action="store_true")
    run_parser.add_argument("--seed", type=int, default=0)

    subparsers.add_parser("status", help="print the current elites and recent programs")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_config(config_path)
    runtime, database_path = runtime_paths(root, config)
    runtime.mkdir(parents=True, exist_ok=True)
    database = ProgramDatabase(database_path)
    try:
        if args.command == "baseline":
            baseline = record_baseline(root, config, args.tier, database)
            print(json.dumps(baseline, indent=2, sort_keys=True))
            return 0
        if args.command == "status":
            print(json.dumps(database.inspirations(config["branch"], limit=20), indent=2, sort_keys=True))
            return 0

        command = proposer_command(config, args.proposer_command)
        if not command:
            raise RuntimeError(
                "no proposer configured; set proposer.command in the config or pass --proposer-command"
            )
        if database.latest_baseline(config["branch"], args.tier) is None:
            record_baseline(root, config, args.tier, database)
        rng = random.Random(args.seed)
        workers = args.parallel or int(config["search"].get("max_parallel", 2))
        temp_root = Path(tempfile.gettempdir()) / "vc2-autoresearch-worktrees"
        accepted: list[dict[str, Any]] = []
        for generation in range(1, args.generations + 1):
            jobs = []
            islands = config["search"].get("islands", ["default"])
            for index in range(args.candidates):
                parent = database.sample_parent(config["branch"], rng, args.tier)
                inspirations = database.inspirations(config["branch"], limit=4)
                jobs.append((parent, inspirations, islands[index % len(islands)]))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        evaluate_candidate,
                        repository=root,
                        config=config,
                        parent=parent,
                        inspirations=inspirations,
                        island=island,
                        generation=generation,
                        command_template=command,
                        tier=args.tier,
                        full_if_promising=args.full_if_promising,
                        temp_root=temp_root,
                    )
                    for parent, inspirations, island in jobs
                ]
                for future in concurrent.futures.as_completed(futures):
                    candidate = future.result()
                    database.add(
                        program_id=candidate["id"],
                        branch=config["branch"],
                        parent_id=candidate["parent_id"],
                        generation=candidate["generation"],
                        island=candidate["island"],
                        commit_sha=candidate["commit_sha"],
                        status=candidate["status"],
                        proposal=candidate["proposal"],
                        metadata=candidate["metadata"],
                        result=candidate["result"],
                    )
                    if candidate["status"] == "accepted":
                        result_tier = candidate["result"].get("tier", args.tier)
                        database.update_elite(
                            config["branch"],
                            f"{result_tier}:{candidate['island']}",
                            candidate["id"],
                        )
                        accepted.append(candidate)
                    print(json.dumps(candidate, indent=2, sort_keys=True))
        if args.promote and accepted:
            best = max(accepted, key=lambda item: database.get(item["id"])["score"])
            promote(root, best)
        return 0
    finally:
        database.close()


if __name__ == "__main__":
    raise SystemExit(main())
