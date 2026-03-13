# IML4Finance Lectures Review Summary

## Files Reviewed
- Lab_01.qmd
- Lab_02.qmd
- Lab_03.qmd
- Lab_04.qmd
- Lecture_01.qmd
- Lecture_02.qmd

## Factual Issues Found and Fixed

### Lecture_02.qmd

**Issue 1: Student Loan Interest Rate Classification**
- **Original text:** `**Student Loans**: Unsecured, term loan, fixed rate`
- **Why it's incorrect:** Federal student loans typically have fixed rates, but private student loans can have either fixed or variable rates. The statement was overly broad.
- **Correction:** Changed to: `**Student Loans**: Unsecured, term loan. Federal student loans typically have fixed rates, while private student loans may have fixed or variable rates.`

**Issue 2: PDO Mathematical Explanation Clarity**
- **Original text:** The scaling factor ($1 / \ln(2)$) ensures that an increase of PDO points in the score exactly halves the odds of default...
- **Why it's unclear:** The explanation of how $B = PDO/\ln(2)$ works was potentially confusing. The factor $1/\ln(2)$ appears in the formula but wasn't clearly connected to why it ensures the doubling/halving property.
- **Correction:** Simplified to: `This factor converts log-odds into score points such that adding PDO points to the score exactly halves the odds of default (or equivalently, doubles the odds of being a "good" borrower).`

## Clarity Issues Reviewed (No Changes Needed)

The following areas were reviewed for clarity but were found to be accurate and well-explained:

### Lecture_01.qmd
- Credit score ranges (FICO 300-850, VantageScore similar range) - **ACCURATE**
- FICO/VantageScore as major scoring models - **ACCURATE**

### Lecture_02.qmd
- Historical dates (FICO 1956 custom models, 1989 general-purpose score, 1995 Fannie Mae/Freddie Mac requirement, 2006 VantageScore) - **ACCURATE**
- PDO mathematical derivation - **VERIFIED CORRECT** (the example calculation shows proper application)
- Credit scoring company vs. credit bureau distinction - **CLEAR AND ACCURATE**

### Lab_01.qmd
- Random Forest probability calibration explanation - **ACCURATE** (Random Forest uses class proportions at leaf nodes, which can be poorly calibrated because Gini impurity doesn't directly optimize probability accuracy)
- CatBoost log loss as proper scoring rule - **ACCURATE**

### Lab_02.qmd
- PD to Score conversion formula - **VERIFIED CORRECT** (matches standard credit scoring formula)
- KS statistic explanation - **ACCURATE**

### Lab_03.qmd
- SHAP value interpretation - **ACCURATE** (the waterfall plot interpretation is correct)
- Counterfactual explanations vs. SHAP distinction - **CLEAR**

### Lab_04.qmd
- ROC-AUC limitations for imbalanced data - **ACCURATE**
- PR-AUC advantages for imbalanced data - **ACCURATE**
- SMOTE explanation - **ACCURATE** (SMOTE needs at least k+1 minority samples where k is the number of neighbors)
- Class imbalance handling strategies - **WELL EXPLAINED**

## Summary Statistics

| File | Factual Issues Fixed | Clarity Issues Fixed |
|------|---------------------|---------------------|
| Lecture_01.qmd | 0 | 0 |
| Lecture_02.qmd | 1 | 1 |
| Lab_01.qmd | 0 | 0 |
| Lab_02.qmd | 0 | 0 |
| Lab_03.qmd | 0 | 0 |
| Lab_04.qmd | 0 | 0 |
| **Total** | **1** | **1** |

## Files Modified
- Lecture_02.qmd (2 edits)

## No Changes Made To
- Lecture_01.qmd
- Lab_01.qmd
- Lab_02.qmd
- Lab_03.qmd
- Lab_04.qmd

## Recommendations for Future Review

1. **Code verification**: While the conceptual explanations were verified, the code examples should be tested to ensure they execute correctly with the provided data.

2. **Data references**: The references to external data files (e.g., `Data/cibil/prospects.parquet`, `Data/lendingclub/`) should be verified to exist in the expected locations.

3. **Helper functions**: The `course_utils/helpers.py` file defines functions like `calculate_score` with default `base_score=600`, but Lab 02 uses `user_basescore = 680`. This inconsistency should be noted for students.

4. **FICO/VantageScore tier boundaries**: The lecture mentions VantageScore has "slightly different tier boundaries" but doesn't specify what they are. Consider adding a footnote or reference for completeness.