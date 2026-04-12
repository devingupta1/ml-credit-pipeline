# Feature Plan — ml-credit-pipeline

## Dataset context
- Shape after merging: 307,511 rows × 224 columns
- Class imbalance: 91.93% class 0 (repaid) / 8.07% class 1 (default)
- Primary metric: **AUC-PR** (never accuracy — 92% accuracy means "predict everyone repays")
- 12 categorical columns, 206 numeric columns, 6 identifier/target columns

## Leakage audit results
- **No LEAK flags** detected (all 206 numeric features have single-feature AUC ≤ 0.70)
- **No INVESTIGATE flags**
- Top features by single-feature AUC (all legitimate):
  - EXT_SOURCE_3: 0.615
  - EXT_SOURCE_2: 0.605
  - BUREAU_DAYS_CREDIT_MEAN: 0.574
  - BUREAU_DAYS_CREDIT_MIN: 0.560
  - DAYS_BIRTH: 0.560
  - EXT_SOURCE_1: 0.559
- Decision: **All features safe to keep** — no drops needed for leakage.

---

## Features to ENGINEER (new columns to create in prompt_07)

### Ratio features
| Name | Formula | Reason |
|---|---|---|
| CREDIT_INCOME_RATIO | AMT_CREDIT / AMT_INCOME_TOTAL | Leverage — how many years of income the loan represents |
| ANNUITY_INCOME_RATIO | AMT_ANNUITY / AMT_INCOME_TOTAL | Payment burden — what fraction of income goes to loan payments |
| CREDIT_TERM | AMT_ANNUITY / AMT_CREDIT | Effective interest proxy — higher ratio = shorter/more expensive loan |
| CREDIT_GOODS_RATIO | AMT_CREDIT / AMT_GOODS_PRICE | Markup over goods price — higher = more financing costs |
| INCOME_PER_PERSON | AMT_INCOME_TOTAL / (CNT_FAM_MEMBERS + 1) | Per-capita income — family size matters for repayment capacity |

### Employment features
| Name | Formula | Reason |
|---|---|---|
| IS_UNEMPLOYED | DAYS_EMPLOYED == 365243 (binary) | Sentinel flag — 18% of applicants, lower default rate (5.4%) |
| DAYS_EMPLOYED_RATIO | DAYS_EMPLOYED / DAYS_BIRTH (exclude sentinel) | Job stability relative to age — longer ratio = more stable |
| EMPLOYMENT_YEARS | abs(DAYS_EMPLOYED) / 365 (set sentinel to NaN first) | Human-readable years employed |

### Age features
| Name | Formula | Reason |
|---|---|---|
| AGE_YEARS | abs(DAYS_BIRTH) / 365 | Human-readable age in years |
| REGISTRATION_AGE_RATIO | DAYS_REGISTRATION / DAYS_BIRTH | How early in life they registered |

### External score features
| Name | Formula | Reason |
|---|---|---|
| EXT_SOURCE_MEAN | mean(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3) | Ensemble of the 3 strongest predictors |
| EXT_SOURCE_STD | std(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3) | Disagreement between scores — high std = conflicting signals |
| EXT_SOURCE_PRODUCT | EXT_SOURCE_1 × EXT_SOURCE_2 × EXT_SOURCE_3 | Interaction — captures joint low-risk profiles |
| EXT_SOURCE_MIN | min(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3) | Worst-case score — bottleneck predictor |

### Cyclical encoding
| Name | Formula | Reason |
|---|---|---|
| WEEKDAY_SIN | sin(2π × weekday_ordinal / 7) | Cyclical encoding preserves proximity (Sunday ≈ Monday) |
| WEEKDAY_COS | cos(2π × weekday_ordinal / 7) | Paired with SIN for full cyclical representation |

### Bureau history features
| Name | Formula | Reason |
|---|---|---|
| BUREAU_ACTIVE_RATIO | BUREAU_ACTIVE_LOAN_COUNT / BUREAU_LOAN_COUNT | Proportion of active vs total bureau loans |
| BUREAU_CREDIT_UTILIZATION | BUREAU_AMT_CREDIT_SUM_DEBT_SUM / BUREAU_AMT_CREDIT_SUM_SUM | How much of approved credit is used |

### Previous application features
| Name | Formula | Reason |
|---|---|---|
| PREV_REFUSAL_RATIO | PREV_REFUSED_COUNT / PREV_LOAN_COUNT | History of being refused — signal of risk |

---

## Features to KEEP as-is

### External scores (top predictors)
- **EXT_SOURCE_1** (56.4% null — keep, median impute, add IS_NULL indicator)
- **EXT_SOURCE_2** (0.2% null — keep, median impute)
- **EXT_SOURCE_3** (keep, median impute)

### Key application features
- DAYS_BIRTH — age proxy, 5th strongest single-feature AUC
- AMT_INCOME_TOTAL — income
- AMT_CREDIT — loan amount
- AMT_ANNUITY — periodic payment
- AMT_GOODS_PRICE — what the loan bought
- REGION_RATING_CLIENT — region risk (1-3)
- REGION_RATING_CLIENT_W_CITY — region+city risk
- REGION_POPULATION_RELATIVE — population density
- DAYS_REGISTRATION — time since registration
- DAYS_ID_PUBLISH — time since ID published
- DAYS_LAST_PHONE_CHANGE — time since phone changed
- CNT_CHILDREN — number of children
- CNT_FAM_MEMBERS — family size

### Flag columns (binary)
- FLAG_OWN_CAR, FLAG_OWN_REALTY — asset ownership
- FLAG_MOBIL, FLAG_EMP_PHONE, FLAG_WORK_PHONE, FLAG_CONT_MOBILE, FLAG_PHONE, FLAG_EMAIL — contact info
- FLAG_DOCUMENT_2 through FLAG_DOCUMENT_21 — document flags
- REG_REGION_NOT_LIVE_REGION, REG_REGION_NOT_WORK_REGION,
  LIVE_REGION_NOT_WORK_REGION, REG_CITY_NOT_LIVE_CITY,
  REG_CITY_NOT_WORK_CITY, LIVE_CITY_NOT_WORK_CITY — address mismatch flags

### Categorical features (to be one-hot or ordinal encoded)
- NAME_CONTRACT_TYPE — Cash vs Revolving
- CODE_GENDER — M/F/XNA
- NAME_INCOME_TYPE — income source
- NAME_EDUCATION_TYPE — education level (ordinal candidate)
- NAME_FAMILY_STATUS — marital status
- NAME_HOUSING_TYPE — housing type
- ORGANIZATION_TYPE — employer type
- OCCUPATION_TYPE — job type (33% null — impute as "Missing")

### Bureau aggregate features (top performers)
- BUREAU_DAYS_CREDIT_MEAN/MIN/STD — credit history length
- BUREAU_LOAN_COUNT, BUREAU_ACTIVE_LOAN_COUNT, BUREAU_CLOSED_LOAN_COUNT
- BUREAU_AMT_CREDIT_SUM_MEAN/SUM — total credit exposure

### Installment features
- INST_LATE_PAYMENT_RATE — 6th strongest single-feature AUC
- INST_DAYS_PAST_DUE — severity of late payments
- INST_PAYMENT_RATIO — payment discipline

### Previous application features
- PREV_LOAN_COUNT, PREV_APPROVED_COUNT, PREV_APPROVAL_RATE
- PREV_DAYS_DECISION_MEAN — how long ago previous decisions

### POS Cash features
- POS_SK_DPD_MEAN/MAX/SUM — payment delays
- POS_COMPLETED_RATE — loan completion history

---

## Features to DROP

### Identifiers (never predictive)
- **SK_ID_CURR** — loan application ID, identifier only

### Raw features replaced by engineered versions
- **DAYS_EMPLOYED** (raw) — replaced by IS_UNEMPLOYED + DAYS_EMPLOYED_RATIO + EMPLOYMENT_YEARS
- **WEEKDAY_APPR_PROCESS_START** (raw categorical) — replaced by WEEKDAY_SIN + WEEKDAY_COS

### High-null housing features (>50% null, redundant AVG/MODE/MEDI triplicates)
Keep only the `_AVG` variant of each housing feature, drop `_MODE` and `_MEDI`:
- Drop: COMMONAREA_MODE, COMMONAREA_MEDI (keep COMMONAREA_AVG)
- Drop: NONLIVINGAPARTMENTS_MODE, NONLIVINGAPARTMENTS_MEDI
- Drop: LIVINGAPARTMENTS_MODE, LIVINGAPARTMENTS_MEDI
- Drop: FLOORSMIN_MODE, FLOORSMIN_MEDI
- Drop: YEARS_BUILD_MODE, YEARS_BUILD_MEDI
- Drop: LANDAREA_MODE, LANDAREA_MEDI
- Drop: BASEMENTAREA_MODE, BASEMENTAREA_MEDI
- Drop: NONLIVINGAREA_MODE, NONLIVINGAREA_MEDI
- Drop: ELEVATORS_MODE, ELEVATORS_MEDI
- Drop: APARTMENTS_MODE, APARTMENTS_MEDI
- Drop: ENTRANCES_MODE, ENTRANCES_MEDI
- Drop: LIVINGAREA_MODE, LIVINGAREA_MEDI
- Drop: FLOORSMAX_MODE, FLOORSMAX_MEDI
- Drop: TOTALAREA_MODE (redundant with _AVG)

> Rationale: AVG/MODE/MEDI are near-identical representations of the same
> Housing info. Keeping all three triples the column count with no new signal.

### Nearly constant features
- FLAG_MOBIL — 99.99% value 1, no discriminative power
- FLAG_DOCUMENT_2, FLAG_DOCUMENT_10, FLAG_DOCUMENT_12, FLAG_DOCUMENT_17,
  FLAG_DOCUMENT_19, FLAG_DOCUMENT_20, FLAG_DOCUMENT_21 — near-zero variance

---

## Features to INVESTIGATE

| Feature | Concern | Action |
|---|---|---|
| OWN_CAR_AGE | 66% null — NaN means no car, not missing data | Keep; impute as 0 if FLAG_OWN_CAR == "N", median if "Y" with null |
| FONDKAPREMONT_MODE | 68% null, categorical | Keep as-is; add "Missing" category |
| BB_DPD_RATE | 90% null — only applicants with bureau balance have this | Keep; null = no bureau balance history, impute with 0 |
| NAME_TYPE_SUITE | Who accompanied the client — unclear predictive value | Keep for now; evaluate after first model |

---

## Imputation strategy

### EXT_SOURCE_* (strongest predictors)
- **EXT_SOURCE_2** (0.2% null): median imputation
- **EXT_SOURCE_3**: median imputation
- **EXT_SOURCE_1** (56.4% null): median imputation + add binary `EXT_SOURCE_1_IS_NULL` indicator
  - High null rate is informative — clients without this score may differ systematically

### Ratio features (division by zero → NaN)
- CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, CREDIT_TERM, etc.: fill with 0 when
  denominator is 0 or NaN

### Bureau aggregates (null = no credit history)
- **Fill with 0**: BUREAU_LOAN_COUNT, BUREAU_ACTIVE_LOAN_COUNT, BUREAU_CLOSED_LOAN_COUNT,
  all count/sum columns — no history means zero loans
- **Fill with median**: BUREAU_DAYS_CREDIT_MEAN, BUREAU_AMT_CREDIT_SUM_MEAN, etc. —
  for averaged stats, 0 would be misleading; median is more neutral
- **Add indicator**: BUREAU_HAS_HISTORY (binary) — whether any bureau records exist

### Previous application aggregates (null = no previous applications)
- Same strategy as bureau: 0 for counts, median for means, add PREV_HAS_HISTORY indicator

### POS Cash / Installment aggregates
- Same strategy: 0 for counts/sums, median for means
- Add POS_HAS_HISTORY and INST_HAS_HISTORY indicators

### High-null housing features (>50%)
- Median imputation + binary IS_NULL indicator for each
- The IS_NULL pattern itself is signal — missing housing data correlates with applicant profile

### Categorical features with missing values
- OCCUPATION_TYPE (33% null): impute as "Missing" category — missingness is informative
- FONDKAPREMONT_MODE, HOUSETYPE_MODE, WALLSMATERIAL_MODE: impute as "Missing" category

---

## Notes

1. **No leakage found** — all 206 numeric features have single-feature AUC ≤ 0.70.
   EXT_SOURCE_3 is the strongest at 0.615 — this is legitimate, not leakage.

2. **Class imbalance strategy** (for prompt_09):
   - Use AUC-PR as primary metric, not accuracy
   - Consider SMOTE or class_weight="balanced" during training
   - The 8.07% positive rate is moderate — not extreme enough to require heavy resampling

3. **DAYS_EMPLOYED sentinel** — 55,374 rows (18%) have value 365243.
   These are unemployed/retired/unknown. Their default rate (5.4%) is actually
   LOWER than the overall rate (8.07%). This is a real signal.

4. **Housing feature triplicates** — the AVG/MODE/MEDI pattern adds 30+ redundant columns.
   Dropping MODE and MEDI variants removes noise without losing signal.

5. **Feature count after engineering**: expect ~180-200 features after drops + new cols.
   This is reasonable for gradient-boosted trees (LightGBM/XGBoost).
