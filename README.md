# Customer-Churn-Prediction-System

Customer churn prediction project with a clean, modular pipeline for:
- Data loading and split
- Model training and evaluation
- Batch inference

## Environment Setup (Conda)

This project uses Conda via `environment.yml`.

```bash
make create-env
conda activate churn-ml
```

If the environment already exists and you changed dependencies:

```bash
make update-env
```

## Packages

Main runtime packages:
- pandas
- numpy
- scikit-learn
- joblib

Development/style packages:
- black
- ruff
- pylint
- mypy

## Quality Checks

Run checks with Makefile:

```bash
make lint
make type-check
```

Or run everything together:

```bash
make check
```

Formatting:

```bash
make format
```

## Useful Commands

```bash
make help
make run
make clean
```

## Google Python Style

This repository is configured to follow Google-style Python conventions:
- Google-style docstrings via Ruff pydocstyle (google convention)
- Strong linting with Ruff and Pylint
- Static typing checks with MyPy
- Consistent formatting with Black

## Project Structure

```text
src/
	data/
		data_loader.py
	models/
		model_trainer.py
		model_evaluator.py
		model_predictor.py
	pipeline/
		training_runner.py
		inference_runner.py
```
