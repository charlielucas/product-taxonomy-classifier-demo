"""Command line entry points for the taxonomy demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from .model import NaiveBayesTextClassifier, accuracy, load_csv, train_test_split, write_csv
from .validation import needs_review, review_reasons


ROOT = Path(__file__).resolve().parents[2]
TRAINING_DATA = ROOT / "data" / "products.csv"
UNLABELED_DATA = ROOT / "data" / "unlabeled_products.csv"
EXAMPLES_DIR = ROOT / "examples"


def build_classifiers(records: list[dict[str, str]]) -> tuple[NaiveBayesTextClassifier, NaiveBayesTextClassifier]:
    category_model = NaiveBayesTextClassifier().fit(records, "category")
    manufacturer_model = NaiveBayesTextClassifier().fit(records, "manufacturer")
    return category_model, manufacturer_model


def evaluate() -> str:
    records = load_csv(TRAINING_DATA)
    train, test = train_test_split(records)
    category_model, manufacturer_model = build_classifiers(train)

    category_accuracy = accuracy(test, category_model, "category")
    manufacturer_accuracy = accuracy(test, manufacturer_model, "manufacturer")

    report = "\n".join(
        [
            "# Model Report",
            "",
            "Synthetic holdout evaluation using every fifth record as the test set.",
            "",
            f"- Training records: {len(train)}",
            f"- Test records: {len(test)}",
            f"- Category accuracy: {category_accuracy:.1%}",
            f"- Manufacturer accuracy: {manufacturer_accuracy:.1%}",
            "",
            "This report is intentionally small. In a production workflow, I would add",
            "per-label precision/recall, drift checks, sampled review, and model versioning.",
            "",
        ]
    )
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    (EXAMPLES_DIR / "model_report.md").write_text(report, encoding="utf-8")
    return report


def predict() -> list[dict[str, object]]:
    records = load_csv(TRAINING_DATA)
    unlabeled = load_csv(UNLABELED_DATA)
    category_model, manufacturer_model = build_classifiers(records)

    rows: list[dict[str, object]] = []
    for record in unlabeled:
        category_prediction = category_model.predict(record)
        manufacturer_prediction = manufacturer_model.predict(record)
        reasons = review_reasons(
            category_prediction.label,
            category_prediction.confidence,
            manufacturer_prediction.label,
            manufacturer_prediction.confidence,
        )
        rows.append(
            {
                **record,
                "predicted_category": category_prediction.label,
                "category_confidence": f"{category_prediction.confidence:.3f}",
                "predicted_manufacturer": manufacturer_prediction.label,
                "manufacturer_confidence": f"{manufacturer_prediction.confidence:.3f}",
                "needs_review": str(needs_review(reasons)),
                "review_reasons": "; ".join(reasons),
            }
        )

    fieldnames = [
        "product_id",
        "vendor",
        "part_number",
        "description",
        "predicted_category",
        "category_confidence",
        "predicted_manufacturer",
        "manufacturer_confidence",
        "needs_review",
        "review_reasons",
    ]
    write_csv(EXAMPLES_DIR / "predictions.csv", rows, fieldnames)
    return rows


def review() -> list[dict[str, object]]:
    rows = predict()
    review_rows = [row for row in rows if row["needs_review"] == "True"]
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(EXAMPLES_DIR / "review_queue.csv", review_rows, fieldnames)
    return review_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic product taxonomy classifier demo")
    parser.add_argument("command", choices=["evaluate", "predict", "review"])
    args = parser.parse_args()

    if args.command == "evaluate":
        print(evaluate())
    elif args.command == "predict":
        rows = predict()
        print(f"Wrote {len(rows)} predictions to {EXAMPLES_DIR / 'predictions.csv'}")
    elif args.command == "review":
        rows = review()
        print(f"Wrote {len(rows)} review rows to {EXAMPLES_DIR / 'review_queue.csv'}")


if __name__ == "__main__":
    main()
