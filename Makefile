PYTHON ?= python3

.PHONY: test evaluate review

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests

evaluate:
	PYTHONPATH=src $(PYTHON) -m product_taxonomy_classifier evaluate

review:
	PYTHONPATH=src $(PYTHON) -m product_taxonomy_classifier review
