# Customer-Churn-Prediction-System

Customer churn prediction project with a clean, modular pipeline for:
- Data loading and split
- Model training and evaluation
- Batch inference

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

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Google Python Style

This repository is configured to follow Google-style Python conventions:
- Google-style docstrings via Ruff pydocstyle (google convention)
- Strong linting with Ruff and Pylint
- Static typing checks with MyPy
- Consistent formatting with Black

Run checks:

```bash
ruff check .
black --check .
pylint src
mypy src
```

## Project Structure

```text
src/
	data/
		load_data.py
	models/
		train.py
		evaluate.py
		predict.py
	pipeline/
		train_pipeline.py
		inference_pipeline.py
```