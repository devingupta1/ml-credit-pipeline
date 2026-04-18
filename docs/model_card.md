# Model card — Home Credit Default Risk

## Model details
- Model type: LightGBM gradient boosting classifier
- Training date: 2026-04-18
- MLflow model: credit-default-risk-champion v1
- Framework: LightGBM

## Intended use
- Primary use: predict probability of loan default for credit risk assessment
- Intended users: credit risk analysts, loan officers
- Out-of-scope uses: any use outside consumer lending context

## Training data
- Dataset: Home Credit Default Risk (Kaggle)
- Training rows: ~246,000 (80% of 307,511)
- Features: 223 after preprocessing pipeline
- Class balance: 91.93% repaid / 8.07% defaulted
- Feature engineering: ratio features, external score aggregation,
  cyclical encoding, sentinel imputation

## Evaluation data
- Validation rows: ~61,500 (20% stratified split)
- Same preprocessing pipeline applied

## Performance
- AUC-PR: 0.9402 (baseline logistic regression: 0.2470)
- AUC-ROC: 0.9954
- Optimal threshold: 0.24 (cost matrix: FN=10x FP)
- Precision at threshold: 73.40%
- Recall at threshold: 97.68%
- F1 at threshold: 83.82%

## Subgroup performance
| Group    | Category        |   AUC-PR |   AUC-ROC |   Default Rate |   Size |
|:---------|:----------------|---------:|----------:|---------------:|-------:|
| Gender   | M               | 0.955914 |  0.995899 |      0.101671  |  20940 |
| Gender   | F               | 0.930391 |  0.99538  |      0.0699194 |  40561 |
| Age      | Young (<30)     | 0.939182 |  0.993064 |      0.11297   |   8905 |
| Age      | Middle (30-50)  | 0.941623 |  0.995291 |      0.0861277 |  31790 |
| Age      | Senior (>=50)   | 0.953987 |  0.997427 |      0.0586794 |  20808 |
| Contract | Cash loans      | 0.942397 |  0.995408 |      0.083507  |  55672 |
| Contract | Revolving loans | 0.925426 |  0.996589 |      0.0541931 |   5831 |

## Top predictive features
1. num__EXT_SOURCE_MEAN: 0.4619
2. cat__CODE_GENDER: 0.1335
3. num__CREDIT_TERM: 0.1276
4. num__AMT_GOODS_PRICE: 0.1036
5. num__AMT_ANNUITY: 0.1028
6. num__DAYS_EMPLOYED_RATIO: 0.0953
7. cat__NAME_EDUCATION_TYPE: 0.0908
8. num__EXT_SOURCE_2: 0.0710
9. num__INST_LATE_PAYMENT_RATE: 0.0682
10. num__BUREAU_AMT_CREDIT_SUM_DEBT_MEAN: 0.0649

## Limitations
- Model trained on historical data from a specific geography and time period
- Performance may degrade on applicant populations not represented in training
- DAYS_EMPLOYED sentinel (18% of applicants) required special handling
- 51 features had >50% null rate — imputed with median

## Fairness findings
Gender: Men (AUC-PR 0.956) vs Women (AUC-PR 0.930) — small gap of ~0.026.
Age: Senior (AUC-PR 0.954) vs Young (AUC-PR 0.939) — small gap of ~0.015.
Contract: Cash loans (AUC-PR 0.942) vs Revolving loans (AUC-PR 0.925) — small gap of ~0.017.
There are no major gaps > 0.05 between demographic groups, indicating the model aligns relatively fairly across subgroups evaluated here.

## Recommendations
- Monitor feature drift monthly (prompt_16 implements this)
- Retrain if AUC-PR drops below 0.22 in production
- Review threshold quarterly as business costs change
