.PHONY: help setup download validate merge eda audit pipeline train evaluate serve test lint monitor clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup:  ## Install deps, hooks, and start MLflow + Postgres
	uv pip install -r requirements.txt
	uv pip install -r requirements-dev.txt
	pre-commit install
	docker compose -f docker/docker-compose.yml up -d
	@echo "Waiting for MLflow to be ready..."
	@sleep 5
	@echo "Setup complete. MLflow UI: http://localhost:5001"

download:  ## Download Kaggle data and register with DVC
	python src/data/ingest.py --download

validate:  ## Run Pandera + Great Expectations validation
	python src/data/validate.py

merge:  ## Aggregate all tables and save merged Parquet
	python src/data/merge.py

eda:  ## Run EDA and generate profiling report + plots
	python src/data/eda.py

audit:  ## Run leakage audit on merged dataset
	python src/data/leakage_audit.py

pipeline:  ## Build and serialize full sklearn pipeline
	python src/features/pipeline.py

train:  ## Train models via src/models/train.py
	.venv/bin/python src/models/train.py

evaluate:  ## Run threshold optimization and calibration
	.venv/bin/python src/models/evaluate.py

serve:  ## Start FastAPI serving on port 8000
	uvicorn src.serving.app:app --reload --port 8000

test:  ## Run pytest with coverage
	.venv/bin/pytest tests/ -v --cov=src --cov-report=term-missing

lint:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

monitor:  ## Run drift detection report
	python src/monitoring/drift_report.py

clean:  ## Stop docker services and remove caches
	docker compose -f docker/docker-compose.yml down
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
