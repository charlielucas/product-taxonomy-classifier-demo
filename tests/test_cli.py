import tempfile
import unittest
from pathlib import Path

from product_taxonomy_classifier.cli import TRAINING_DATA, UNLABELED_DATA, evaluate, review


class CliTests(unittest.TestCase):
    def test_evaluate_writes_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            report = evaluate(training_data=TRAINING_DATA, output_dir=output_dir)

            self.assertIn("Category accuracy", report)
            self.assertTrue((output_dir / "model_report.md").exists())

    def test_review_threshold_changes_queue_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            strict_rows = review(
                training_data=TRAINING_DATA,
                unlabeled_data=UNLABELED_DATA,
                output_dir=Path(tmpdir),
                threshold=0.95,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            relaxed_rows = review(
                training_data=TRAINING_DATA,
                unlabeled_data=UNLABELED_DATA,
                output_dir=Path(tmpdir),
                threshold=0.50,
            )

        self.assertGreater(len(strict_rows), len(relaxed_rows))


if __name__ == "__main__":
    unittest.main()
