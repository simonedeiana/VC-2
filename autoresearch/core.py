"""Core primitives for the VC-2 evolutionary optimization loop."""

from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


BLOCK_RE = re.compile(
    r"(?ms)^FILE:\s*(?P<path>[^\r\n]+)\r?\n"
    r"<<<<<<< SEARCH\r?\n(?P<search>.*?)"
    r"^=======\r?\n(?P<replace>.*?)"
    r"^>>>>>>> REPLACE\s*$"
)
META_RE = re.compile(r"(?mi)^(TARGET|STRATEGY|RISK):\s*(.+?)\s*$")


@dataclass(frozen=True)
class Edit:
    path: str
    search: str
    replace: str


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"name", "branch", "evaluation", "search"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"missing config keys: {', '.join(missing)}")
    return config


def parse_proposal(text: str) -> tuple[dict[str, str], list[Edit]]:
    metadata = {key.lower(): value.strip() for key, value in META_RE.findall(text)}
    edits = [
        Edit(
            path=match.group("path").strip().replace("\\", "/"),
            search=match.group("search"),
            replace=match.group("replace"),
        )
        for match in BLOCK_RE.finditer(text)
    ]
    if not edits:
        raise ValueError("proposal contains no FILE/SEARCH/REPLACE blocks")
    return metadata, edits


def apply_edits(root: Path, edits: Iterable[Edit]) -> list[str]:
    root = root.resolve()
    touched: list[str] = []
    for edit in edits:
        target = (root / edit.path).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"proposal escapes repository: {edit.path}") from exc
        if not target.is_file():
            raise ValueError(f"proposal target does not exist: {edit.path}")
        content = target.read_text(encoding="utf-8")
        occurrences = content.count(edit.search)
        if occurrences != 1:
            raise ValueError(
                f"SEARCH block for {edit.path} matched {occurrences} times; expected exactly one"
            )
        target.write_text(content.replace(edit.search, edit.replace, 1), encoding="utf-8", newline="")
        touched.append(edit.path)
    return sorted(set(touched))


def median(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot calculate a median from no values")
    return float(statistics.median(values))


def geometric_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values or any(value <= 0 for value in values):
        return 0.0
    return float(statistics.geometric_mean(values))


def performance_metrics(result: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in result.get("metrics", {}).items()
        if key.endswith("_fps") and isinstance(value, (int, float))
    }


def aggregate_score(result: dict[str, Any]) -> float:
    return geometric_mean(performance_metrics(result).values())


def correctness_matches(candidate: dict[str, Any], parent: dict[str, Any]) -> tuple[bool, str]:
    candidate_hashes = candidate.get("hashes", {})
    parent_hashes = parent.get("hashes", {})
    if not parent_hashes:
        return True, "parent has no correctness hashes"
    missing = sorted(set(parent_hashes).difference(candidate_hashes))
    if missing:
        return False, f"missing hashes: {', '.join(missing)}"
    mismatched = sorted(key for key in parent_hashes if candidate_hashes[key] != parent_hashes[key])
    if mismatched:
        return False, f"output changed: {', '.join(mismatched)}"
    return True, "byte-identical to parent"


def promising(
    candidate: dict[str, Any],
    parent: dict[str, Any],
    minimum_gain_percent: float,
    regression_tolerance_percent: float,
) -> tuple[bool, str]:
    child = performance_metrics(candidate)
    base = performance_metrics(parent)
    common = sorted(set(child).intersection(base))
    if not common:
        return False, "no comparable performance metrics"
    gains = {key: 100.0 * (child[key] / base[key] - 1.0) for key in common if base[key] > 0}
    if any(value < -regression_tolerance_percent for value in gains.values()):
        worst = min(gains, key=gains.get)
        return False, f"{worst} regressed {gains[worst]:.2f}%"
    if max(gains.values(), default=float("-inf")) < minimum_gain_percent:
        return False, f"best gain was {max(gains.values(), default=0.0):.2f}%"
    summary = ", ".join(f"{key}={value:+.2f}%" for key, value in gains.items())
    return True, summary


def candidate_cell(metadata: dict[str, str], islands: list[str]) -> str:
    target = metadata.get("target", "").lower()
    strategy = metadata.get("strategy", "").lower()
    combined = f"{target}-{strategy}"
    for island in islands:
        tokens = [token for token in island.lower().split("-") if token]
        if tokens and all(token in combined for token in tokens):
            return island
    digest = int(hashlib.sha256(combined.encode("utf-8")).hexdigest()[:8], 16)
    return islands[digest % len(islands)] if islands else "default"


class ProgramDatabase:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS programs (
                id TEXT PRIMARY KEY,
                branch TEXT NOT NULL,
                parent_id TEXT,
                generation INTEGER NOT NULL,
                island TEXT NOT NULL,
                commit_sha TEXT,
                status TEXT NOT NULL,
                proposal TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                score REAL NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS programs_branch_score
                ON programs(branch, status, score DESC);
            CREATE TABLE IF NOT EXISTS elites (
                branch TEXT NOT NULL,
                island TEXT NOT NULL,
                program_id TEXT NOT NULL,
                PRIMARY KEY(branch, island),
                FOREIGN KEY(program_id) REFERENCES programs(id)
            );
            """
        )

    def close(self) -> None:
        self.connection.close()

    def add(
        self,
        *,
        program_id: str,
        branch: str,
        parent_id: str | None,
        generation: int,
        island: str,
        commit_sha: str | None,
        status: str,
        proposal: str,
        metadata: dict[str, str],
        result: dict[str, Any],
    ) -> None:
        score = aggregate_score(result) if result.get("valid") else 0.0
        self.connection.execute(
            """INSERT OR REPLACE INTO programs
               (id, branch, parent_id, generation, island, commit_sha, status,
                proposal, metadata_json, result_json, score, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                program_id,
                branch,
                parent_id,
                generation,
                island,
                commit_sha,
                status,
                proposal,
                json.dumps(metadata, sort_keys=True),
                json.dumps(result, sort_keys=True),
                score,
                time.time(),
            ),
        )
        self.connection.commit()

    def get(self, program_id: str) -> dict[str, Any]:
        row = self.connection.execute("SELECT * FROM programs WHERE id = ?", (program_id,)).fetchone()
        if row is None:
            raise KeyError(program_id)
        return self._decode(row)

    def latest_baseline(self, branch: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """SELECT * FROM programs
               WHERE branch = ? AND status = 'baseline'
               ORDER BY created_at DESC LIMIT 1""",
            (branch,),
        ).fetchone()
        return self._decode(row) if row else None

    def inspirations(self, branch: str, limit: int = 4) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT * FROM programs
               WHERE branch = ? AND status IN ('baseline', 'accepted')
               ORDER BY score DESC, created_at DESC LIMIT ?""",
            (branch, limit),
        ).fetchall()
        return [self._decode(row) for row in rows]

    def sample_parent(self, branch: str, rng: random.Random) -> dict[str, Any]:
        rows = self.connection.execute(
            """SELECT p.* FROM elites e JOIN programs p ON p.id = e.program_id
               WHERE e.branch = ?
               UNION
               SELECT * FROM programs WHERE branch = ? AND status = 'baseline'
               ORDER BY score DESC""",
            (branch, branch),
        ).fetchall()
        if not rows:
            raise RuntimeError(f"no baseline or elites recorded for {branch}")
        population = [self._decode(row) for row in rows]
        weights = [max(item["score"], 1.0) for item in population]
        return rng.choices(population, weights=weights, k=1)[0]

    def update_elite(self, branch: str, island: str, program_id: str) -> bool:
        candidate = self.get(program_id)
        row = self.connection.execute(
            """SELECT p.score FROM elites e JOIN programs p ON p.id = e.program_id
               WHERE e.branch = ? AND e.island = ?""",
            (branch, island),
        ).fetchone()
        if row is not None and float(row["score"]) >= candidate["score"]:
            return False
        self.connection.execute(
            """INSERT INTO elites(branch, island, program_id) VALUES (?, ?, ?)
               ON CONFLICT(branch, island) DO UPDATE SET program_id = excluded.program_id""",
            (branch, island, program_id),
        )
        self.connection.commit()
        return True

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json"))
        item["result"] = json.loads(item.pop("result_json"))
        return item
