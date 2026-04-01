SHELL := /bin/bash

ENV_FILE := environment.yml
ENV_NAME := churn-ml

.PHONY: create-env update-env remove-env format lint type-check check run clean

create-env:
	conda env create -f $(ENV_FILE)

update-env:
	conda env update -f $(ENV_FILE) --prune

remove-env:
	conda env remove -n $(ENV_NAME)

format:
	black .

lint:
	ruff check .
	pylint src

type-check:
	mypy src

check: lint type-check

run:
	python app.py

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
