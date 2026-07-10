"""Small multinomial Naive Bayes text classifier.

The goal is not to beat a mature ML library. It is to keep the workflow easy
to inspect: train on labeled records, score new records, and expose confidence
for review.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .text import record_text, tokenize


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    scores: dict[str, float]


class NaiveBayesTextClassifier:
    def __init__(self) -> None:
        self.labels: set[str] = set()
        self.label_counts: Counter[str] = Counter()
        self.token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.total_tokens: Counter[str] = Counter()
        self.vocabulary: set[str] = set()
        self.trained = False

    def fit(self, records: list[dict[str, str]], target: str) -> "NaiveBayesTextClassifier":
        for record in records:
            label = record[target].strip()
            if not label:
                continue
            tokens = tokenize(record_text(record))
            self.labels.add(label)
            self.label_counts[label] += 1
            self.token_counts[label].update(tokens)
            self.total_tokens[label] += len(tokens)
            self.vocabulary.update(tokens)
        if not self.labels:
            raise ValueError(f"No labels found for target {target!r}")
        self.trained = True
        return self

    def predict(self, record: dict[str, str]) -> Prediction:
        if not self.trained:
            raise RuntimeError("Classifier must be fitted before prediction")

        tokens = tokenize(record_text(record))
        vocab_size = max(len(self.vocabulary), 1)
        total_docs = sum(self.label_counts.values())
        log_scores: dict[str, float] = {}

        for label in sorted(self.labels):
            prior = math.log(self.label_counts[label] / total_docs)
            denom = self.total_tokens[label] + vocab_size
            token_score = 0.0
            for token in tokens:
                token_score += math.log((self.token_counts[label][token] + 1) / denom)
            log_scores[label] = prior + token_score

        probabilities = softmax(log_scores)
        label = max(probabilities, key=probabilities.get)
        return Prediction(label=label, confidence=probabilities[label], scores=probabilities)


def softmax(log_scores: dict[str, float]) -> dict[str, float]:
    max_score = max(log_scores.values())
    exp_scores = {label: math.exp(score - max_score) for label, score in log_scores.items()}
    total = sum(exp_scores.values())
    return {label: value / total for label, value in exp_scores.items()}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_test_split(records: list[dict[str, str]], every_nth: int = 5) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for index, record in enumerate(records):
        if index % every_nth == 0:
            test.append(record)
        else:
            train.append(record)
    return train, test


def accuracy(records: list[dict[str, str]], classifier: NaiveBayesTextClassifier, target: str) -> float:
    if not records:
        return 0.0
    correct = 0
    for record in records:
        prediction = classifier.predict(record)
        if prediction.label == record[target]:
            correct += 1
    return correct / len(records)
