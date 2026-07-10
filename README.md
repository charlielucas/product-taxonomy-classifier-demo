# Product Taxonomy Classifier Demo

A small synthetic demo of a product taxonomy workflow: train a text classifier, score unlabeled product records, flag low-confidence predictions, and create a review queue for human validation.

This is not employer code and does not use real company data. The products, vendors, labels, and manufacturers are synthetic.

## Why I Built This

I wanted a safe public version of a workflow I care about: turning inconsistent product descriptions into structured labels that people can inspect and improve. The useful part is not only making predictions. It is making the output reviewable enough that a teammate can see what the model thinks, where it is unsure, and what needs a human decision.

## What It Shows

- A lightweight text classifier written in Python with no external ML dependencies
- Multi-field product records with vendor, part number, and description text
- Category and manufacturer prediction
- Confidence scoring
- Review queue creation for low-confidence records
- Validation checks for allowed labels
- A short model report for practical inspection

## Project Structure

```text
product-taxonomy-classifier-demo/
  data/
    products.csv
    unlabeled_products.csv
  examples/
    model_report.md
    review_queue.csv
  src/
    product_taxonomy_classifier/
      cli.py
      model.py
      text.py
      validation.py
  tests/
    test_model.py
    test_validation.py
```

## Quick Start

Use Python 3.10 or newer.

```bash
python -m product_taxonomy_classifier.cli evaluate
python -m product_taxonomy_classifier.cli predict
python -m product_taxonomy_classifier.cli review
```

If running from a fresh clone, set `PYTHONPATH`:

```bash
PYTHONPATH=src python -m product_taxonomy_classifier.cli evaluate
PYTHONPATH=src python -m product_taxonomy_classifier.cli predict
PYTHONPATH=src python -m product_taxonomy_classifier.cli review
```

Run tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Example Output

The review command writes:

- `examples/review_queue.csv`
- `examples/model_report.md`

The queue is meant for a human reviewer. It includes the original record, predicted labels, confidence scores, and review reasons.

## Design Notes

This repo intentionally uses a small pure-Python multinomial Naive Bayes classifier. A production workflow would likely use scikit-learn, embeddings, richer model evaluation, and stronger monitoring, but the small implementation keeps the logic easy to inspect.

The important workflow shape is:

1. Train on known records.
2. Predict labels for new records.
3. Validate labels against allowed taxonomy values.
4. Flag low-confidence or invalid outputs.
5. Send uncertain records to a review queue.

## Safety Notes

- No real employer data
- No internal schema names
- No private business records or real operational data
- No credentials
- No API keys
- No private work code

## Possible Next Steps

- Add a small React review UI
- Add richer evaluation metrics
- Add grouped reporting by vendor
- Add a mock LLM review step for low-confidence predictions
- Add model/version metadata to each output file
