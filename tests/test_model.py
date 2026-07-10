import unittest

from product_taxonomy_classifier.model import NaiveBayesTextClassifier


class ModelTests(unittest.TestCase):
    def test_predicts_label_with_confidence(self):
        records = [
            {"vendor": "A", "part_number": "BRK-1", "description": "front brake pad rotor", "category": "Brake System"},
            {"vendor": "A", "part_number": "BRK-2", "description": "rear brake caliper hose", "category": "Brake System"},
            {"vendor": "B", "part_number": "CAB-1", "description": "cabin air filter pollen", "category": "Cabin Comfort"},
            {"vendor": "B", "part_number": "CAB-2", "description": "blower motor resistor hvac", "category": "Cabin Comfort"},
        ]
        model = NaiveBayesTextClassifier().fit(records, "category")
        prediction = model.predict({"vendor": "A", "part_number": "BRK-9", "description": "brake pad sensor"})

        self.assertEqual(prediction.label, "Brake System")
        self.assertGreater(prediction.confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
