# ml-credit-pipeline

End-to-end tabular ML pipeline for Home Credit Default Risk prediction.

## Setup

```bash
git clone <repo-url> ml-credit-pipeline
cd ml-credit-pipeline
make setup
make train
make serve
```

## Architecture

[Architecture diagram — added in step 17]

## Usage

TODO

## Results

TODO

## Project structure

```
ml-credit-pipeline/
├── .python-version
├── .gitignore
├── README.md
├── requirements.txt
├── requirements-dev.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   └── validate.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── transformers.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── serving/
│       ├── __init__.py
│       ├── app.py
│       ├── schemas.py
│       └── predictor.py
│
├── tests/
│   ├── __init__.py
│   ├── test_transformers.py
│   └── test_serving.py
│
├── models/
│
├── reports/
│
├── docs/
│   └── model_card.md
│
└── docker/
    └── .gitkeep
```
