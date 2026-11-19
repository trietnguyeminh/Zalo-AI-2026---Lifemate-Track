from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lifemate_portfolio.artifacts import atomic_json, fingerprint, file_sha256, verify_receipt


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.contract = {"model": "synthetic", "split": "fit", "beams": 1}
        (self.root / "weights.fixture").write_bytes(b"synthetic bytes, not weights")
        self.receipt = {"complete": True, "contract_sha256": fingerprint(self.contract),
                        "files": {"weights.fixture": file_sha256(self.root / "weights.fixture")}}

    def test_contract_order_is_canonical(self):
        self.assertEqual(fingerprint(self.contract), fingerprint(dict(reversed(list(self.contract.items())))))

    def test_valid_receipt(self):
        verify_receipt(self.root, self.receipt, self.contract)

    def test_split_decode_or_model_change_invalidates_cache(self):
        for changed in ({"split": "dev"}, {"beams": 5}, {"model": "another"}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                verify_receipt(self.root, self.receipt, self.contract | changed)

    def test_changed_file_is_rejected(self):
        (self.root / "weights.fixture").write_bytes(b"interrupted replacement")
        with self.assertRaises(ValueError):
            verify_receipt(self.root, self.receipt, self.contract)

    def test_missing_file_is_rejected(self):
        (self.root / "weights.fixture").unlink()
        with self.assertRaises(ValueError):
            verify_receipt(self.root, self.receipt, self.contract)

    def test_incomplete_or_empty_receipt_is_rejected(self):
        for change in ({"complete": False}, {"files": {}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify_receipt(self.root, self.receipt | change, self.contract)

    def test_path_traversal_and_state_paths_rejected_before_read(self):
        for name in ("../escape", "/absolute", "a\\b", "sample_environment/state.json", "state.json"):
            bad = self.receipt | {"files": {name: "not_a_hash"}}
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify_receipt(self.root, bad, self.contract)

    def test_failed_publication_preserves_old_manifest(self):
        path = self.root / "receipt.json"
        atomic_json(path, {"generation": "complete_old"})
        before = path.read_bytes()
        with patch("lifemate_portfolio.artifacts.os.replace", side_effect=OSError("injected failure")):
            with self.assertRaises(OSError):
                atomic_json(path, {"generation": "new"})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.root.glob("*.partial")), [])

    def test_nonfinite_metadata_not_published(self):
        path = self.root / "receipt.json"
        with self.assertRaises(ValueError):
            atomic_json(path, {"loss": float("nan")})
        self.assertFalse(path.exists())
