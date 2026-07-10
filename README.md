# Product Taxonomy Classifier

A small Python project for classifying product records into a controlled taxonomy.

The data is synthetic. It is not employer code, and it does not use real customer, vendor, or product data.

## What It Does

The workflow trains a lightweight text classifier on labeled product records, scores new records, and writes a review queue for records that need a closer look.

The repo is small, but it covers the main pieces of the workflow:

- combine vendor, part number, and description text
- predict product category and manufacturer
- keep confidence scores with each prediction
- validate labels against allowed values
- flag lower-confidence records for review
- write simple outputs that can be checked by a person

## Project Structure

```text
product-taxonomy-classifier-demo/
  data/
    products.csv
    unlabeled_products.csv
  examples/
    model_report.md
    predictions.csv
    review_queue.csv
  src/
    product_taxonomy_classifier/
      __main__.py
      cli.py
      model.py
      text.py
      validation.py
  tests/
    test_cli.py
    test_model.py
    test_validation.py
  Makefile
```

## Quick Start

Use Python 3.9 or newer.

```bash
PYTHONPATH=src python3 -m product_taxonomy_classifier evaluate
PYTHONPATH=src python3 -m product_taxonomy_classifier predict
PYTHONPATH=src python3 -m product_taxonomy_classifier review
```

Or use the Makefile:

```bash
make test
make evaluate
make review
```

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Command Options

The review threshold can be tuned without changing code:

```bash
PYTHONPATH=src python3 -m product_taxonomy_classifier review --threshold 0.85
```

You can also pass alternate input and output paths:

```bash
PYTHONPATH=src python3 -m product_taxonomy_classifier predict \
  --training-data data/products.csv \
  --unlabeled-data data/unlabeled_products.csv \
  --output-dir examples
```

## Outputs

The commands write:

- `examples/model_report.md`
- `examples/predictions.csv`
- `examples/review_queue.csv`

The review queue keeps the original record next to the predicted labels, confidence scores, and review reasons.

## Design Notes

The classifier is a small pure-Python multinomial Naive Bayes model. That keeps the logic easy to inspect in a short repo.

A larger version of this would probably use scikit-learn or embeddings, add better evaluation, and track model versions. For this repo, the goal is to show the workflow clearly without hiding the important parts behind a library call.

## Known Limits

- The dataset is small and synthetic.
- The model report uses a simple holdout split.
- Confidence scores are useful for review routing, not calibration.
- The taxonomy is fixed in code for this demo.

