import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import (  # noqa: E402
    ProgramDatabase,
    aggregate_score,
    apply_edits,
    candidate_cell,
    correctness_matches,
    parse_proposal,
    promising,
)


class ProposalTests(unittest.TestCase):
    def test_parse_and_apply(self):
        proposal = """TARGET: decoder
STRATEGY: test-change
RISK: low
FILE: source.cpp
<<<<<<< SEARCH
return 1;
=======
return 2;
>>>>>>> REPLACE
"""
        metadata, edits = parse_proposal(proposal)
        self.assertEqual(metadata["target"], "decoder")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "source.cpp"
            target.write_text("int f() {\nreturn 1;\n}\n", encoding="utf-8")
            self.assertEqual(apply_edits(root, edits), ["source.cpp"])
            self.assertIn("return 2;", target.read_text(encoding="utf-8"))

    def test_ambiguous_search_is_rejected(self):
        proposal = """FILE: source.cpp
<<<<<<< SEARCH
x
=======
y
>>>>>>> REPLACE
"""
        _, edits = parse_proposal(proposal)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.cpp").write_text("x\nx\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "matched 2 times"):
                apply_edits(root, edits)

    def test_path_escape_is_rejected(self):
        proposal = """FILE: ../outside.cpp
<<<<<<< SEARCH
x
=======
y
>>>>>>> REPLACE
"""
        _, edits = parse_proposal(proposal)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "escapes repository"):
                apply_edits(Path(directory), edits)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.parent = {
            "valid": True,
            "metrics": {"encoder_fps": 100.0, "decoder_fps": 200.0},
            "hashes": {"stream": "abc", "pixels": "def"},
        }

    def test_correctness_and_multi_metric_selection(self):
        child = {
            "valid": True,
            "metrics": {"encoder_fps": 101.0, "decoder_fps": 200.2},
            "hashes": dict(self.parent["hashes"]),
        }
        self.assertTrue(correctness_matches(child, self.parent)[0])
        self.assertTrue(promising(child, self.parent, 0.5, 0.5)[0])
        self.assertGreater(aggregate_score(child), aggregate_score(self.parent))

    def test_regression_blocks_candidate(self):
        child = {
            "valid": True,
            "metrics": {"encoder_fps": 102.0, "decoder_fps": 198.0},
            "hashes": dict(self.parent["hashes"]),
        }
        accepted, reason = promising(child, self.parent, 0.5, 0.5)
        self.assertFalse(accepted)
        self.assertIn("regressed", reason)

    def test_island_mapping(self):
        islands = ["encoder-transform", "decoder-entropy"]
        self.assertEqual(
            candidate_cell({"target": "encoder", "strategy": "transform-fusion"}, islands),
            "encoder-transform",
        )


class DatabaseTests(unittest.TestCase):
    def test_program_and_elite_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            database = ProgramDatabase(Path(directory) / "programs.sqlite3")
            result = {
                "valid": True,
                "tier": "quick",
                "metrics": {"encoder_fps": 10.0},
                "hashes": {},
            }
            database.add(
                program_id="baseline",
                branch="optimizations",
                parent_id=None,
                generation=0,
                island="baseline",
                commit_sha="abc",
                status="baseline",
                proposal="baseline",
                metadata={},
                result=result,
            )
            self.assertTrue(database.update_elite("optimizations", "encoder-transform", "baseline"))
            self.assertEqual(
                database.sample_parent("optimizations", __import__("random").Random(0), "quick")["id"],
                "baseline",
            )
            self.assertIsNone(database.latest_baseline("optimizations", "full"))
            database.close()


if __name__ == "__main__":
    unittest.main()
