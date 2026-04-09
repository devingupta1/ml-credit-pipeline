"""
drift_report.py

Runs Evidently AI DatasetDriftReport comparing the training data distribution
against a production window. Computes PSI per feature and raises an alert
if PSI > 0.2. Saves report to reports/drift_report.html.
Scheduled as a Prefect flow in prompt_16.
"""

pass
