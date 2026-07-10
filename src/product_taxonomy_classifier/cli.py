"""Command line entry points for the taxonomy workflow."""

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


def label_accuracy_rows(
    records: list[dict[str, str]],
    classifier: NaiveBayesTextClassifier,
    target: str,
) -> list[str]:
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}

    for record in records:
        expected = record[target]
        totals[expected] = totals.get(expected, 0) + 1
        if classifier.predict(record).label == expected:
            correct[expected] = correct.get(expected, 0) + 1

    rows = []
    for label in sorted(totals):
        label_correct = correct.get(label, 0)
        rows.append(f"- {label}: {label_correct}/{totals[label]}")
    return rows


def evaluate(training_data: Path = TRAINING_DATA, output_dir: Path = EXAMPLES_DIR) -> str:
    records = load_csv(training_data)
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
            "## Category Checks",
            "",
            *label_accuracy_rows(test, category_model, "category"),
            "",
            "## Manufacturer Checks",
            "",
            *label_accuracy_rows(test, manufacturer_model, "manufacturer"),
            "",
            "The holdout is small by design. The report is meant to show the workflow",
            "shape rather than claim production model quality.",
            "",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_report.md").write_text(report, encoding="utf-8")
    return report


def predict(
    training_data: Path = TRAINING_DATA,
    unlabeled_data: Path = UNLABELED_DATA,
    output_dir: Path = EXAMPLES_DIR,
    threshold: float = 0.72,
) -> list[dict[str, object]]:
    records = load_csv(training_data)
    unlabeled = load_csv(unlabeled_data)
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
            threshold=threshold,
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
    write_csv(output_dir / "predictions.csv", rows, fieldnames)
    return rows


def review(
    training_data: Path = TRAINING_DATA,
    unlabeled_data: Path = UNLABELED_DATA,
    output_dir: Path = EXAMPLES_DIR,
    threshold: float = 0.72,
) -> list[dict[str, object]]:
    rows = predict(
        training_data=training_data,
        unlabeled_data=unlabeled_data,
        output_dir=output_dir,
        threshold=threshold,
    )
    review_rows = [row for row in rows if row["needs_review"] == "True"]
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(output_dir / "review_queue.csv", review_rows, fieldnames)
    return review_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Product taxonomy classification workflow")
    parser.add_argument("command", choices=["evaluate", "predict", "review"])
    parser.add_argument("--training-data", type=Path, default=TRAINING_DATA)
    parser.add_argument("--unlabeled-data", type=Path, default=UNLABELED_DATA)
    parser.add_argument("--output-dir", type=Path, default=EXAMPLES_DIR)
    parser.add_argument("--threshold", type=float, default=0.72)
    args = parser.parse_args()

    if args.command == "evaluate":
        print(evaluate(training_data=args.training_data, output_dir=args.output_dir))
    elif args.command == "predict":
        rows = predict(
            training_data=args.training_data,
            unlabeled_data=args.unlabeled_data,
            output_dir=args.output_dir,
            threshold=args.threshold,
        )
        print(f"Wrote {len(rows)} predictions to {args.output_dir / 'predictions.csv'}")
    elif args.command == "review":
        rows = review(
            training_data=args.training_data,
            unlabeled_data=args.unlabeled_data,
            output_dir=args.output_dir,
            threshold=args.threshold,
        )
        print(f"Wrote {len(rows)} review rows to {args.output_dir / 'review_queue.csv'}")


if __name__ == "__main__":
    main()
