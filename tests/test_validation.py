import unittest

from product_taxonomy_classifier.validation import needs_review, review_reasons


class ValidationTests(unittest.TestCase):
    def test_low_confidence_prediction_needs_review(self):
        reasons = review_reasons(
            category="Brake System",
            category_confidence=0.51,
            manufacturer="Apex AutoWorks",
            manufacturer_confidence=0.91,
        )

        self.assertIn("low category confidence", reasons)
        self.assertTrue(needs_review(reasons))

    def test_allowed_high_confidence_prediction_does_not_need_review(self):
        reasons = review_reasons(
            category="Brake System",
            category_confidence=0.95,
            manufacturer="Apex AutoWorks",
            manufacturer_confidence=0.91,
        )

        self.assertEqual(reasons, [])
        self.assertFalse(needs_review(reasons))

    def test_threshold_can_be_tuned(self):
        strict_reasons = review_reasons(
            category="Brake System",
            category_confidence=0.80,
            manufacturer="Apex AutoWorks",
            manufacturer_confidence=0.91,
            threshold=0.85,
        )
        relaxed_reasons = review_reasons(
            category="Brake System",
            category_confidence=0.80,
            manufacturer="Apex AutoWorks",
            manufacturer_confidence=0.91,
            threshold=0.70,
        )

        self.assertIn("low category confidence", strict_reasons)
        self.assertEqual(relaxed_reasons, [])


if __name__ == "__main__":
    unittest.main()
